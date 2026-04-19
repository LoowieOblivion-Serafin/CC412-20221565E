"""
===============================================================================
FASE 2 — PIPELINE DE INFERENCIA SD 2.1 unCLIP (integración end-to-end)
===============================================================================

Proyecto ACECOM — Reconstrucción de imágenes mentales desde fMRI.
Base: Koide-Majima et al. (2024). Plan en MIGRATION.md §6 (config A1/A2).

QUÉ HACE ESTE SCRIPT
--------------------
1. Carga los features CLIP predichos del cerebro (mismos .pkl que alimentaban
   VQGAN en Fase 1) para S01, S02, S03.
2. Carga el pipeline SD 2.1 unCLIP (sd_decoder.py).
3. Reemplaza el scheduler por default por DPMSolverMultistepScheduler (25 pasos).
4. Fija seed global + seed por-llamada para reproducibilidad bit-exacta.
5. Itera cada (sujeto, imagen), reconstruye vía `reconstruct_from_embedding`,
   guarda PNG en `output_sd_reconstructions/SXX/`.

EJECUCIÓN AISLADA
-----------------
    python phase2_run_sd.py                 # los 3 sujetos, todas las imágenes
    python phase2_run_sd.py --subjects S01  # sólo S01
    python phase2_run_sd.py --limit 5       # sólo 5 imágenes por sujeto (smoke)
    python phase2_run_sd.py --cpu           # forzar CPU (debugging)

DIMENSIONALIDAD — NOTA CRÍTICA
------------------------------
Los features del dataset actual son **CLIP ViT-B/32 (512-d)**, porque el
pipeline de Fase 1 predijo contra ese encoder. SD 2.1 unCLIP espera
**CLIP ViT-L/14 (768-d)**. Este script implementa un PUENTE TEMPORAL
(zero-pad 512→768) sólo para validar la plumbing end-to-end.

La solución definitiva es el **adapter fMRI→CLIP ViT-L/14** (Fase 2 Sem 3-4
según MIGRATION.md §7). Hasta entonces, las reconstrucciones que produce
este script son un SMOKE TEST de la integración, NO el experimento final.
Verás una advertencia explícita en cada corrida si el shim está activo.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler

# ---- Imports del proyecto ---------------------------------------------------
# Hack de path (igual que main_local_decoder.py) para que el módulo corra
# invocado directamente desde la raíz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from sd_decoder import (
    load_sd_unclip_pipeline,
    reconstruct_from_embedding,
    SD_PRIOR_PROMPTS,
)

# ============================================================================
# CONSTANTES
# ============================================================================

GLOBAL_SEED = 42
INFERENCE_STEPS = 25
GUIDANCE_SCALE = 10.0
OUTPUT_ROOT = config.PROJECT_ROOT / "output_sd_reconstructions"
UNCLIP_EMBED_DIM = 768  # CLIP ViT-L/14 — dimensión que SD 2.1 unCLIP requiere

logger = logging.getLogger("phase2_run_sd")


# ============================================================================
# CARGA DE FEATURES (subset CLIP-only de read_decoded_features de Fase 1)
# ============================================================================

def load_clip_features_for_subject(subject_id: str) -> dict[str, torch.Tensor]:
    """
    Lee sólo los features CLIP predichos del cerebro para un sujeto.

    Estructura esperada (idéntica a Fase 1):
        features/decoded_features/{subject_id}/CLIP_ViT-B_32/lastLayer/imagery__*.pkl

    A diferencia de `main_local_decoder.read_decoded_features`, aquí NO
    leemos VGG — SD unCLIP sólo usa la condición CLIP. Esto acelera el
    arranque del script ~10× (menos I/O).

    Returns:
        dict {image_id: tensor (512,) normalizado L2}
    """
    clip_dir = config.get_clip_path(subject_id)
    if not clip_dir.exists():
        logger.error(f"[{subject_id}] Directorio CLIP no encontrado: {clip_dir}")
        return {}

    all_files = [f.name for f in clip_dir.iterdir() if f.is_file()]
    valid = config.filter_valid_files(
        all_files,
        ignore_hidden=config.PROCESSING_CONFIG["ignore_hidden_files"],
        required_substring=config.PROCESSING_CONFIG["required_substring"],
    )

    out: dict[str, torch.Tensor] = {}
    for pkl_file in valid:
        if not pkl_file.endswith(".pkl"):
            continue
        try:
            with open(clip_dir / pkl_file, "rb") as f:
                data = pickle.load(f)
            image_id = pkl_file.replace("imagery__", "").replace(".pkl", "")

            if isinstance(data, dict):
                clip_feat = data.get("feat", data.get("features", data))
            else:
                clip_feat = data

            clip_feat = torch.tensor(clip_feat, dtype=torch.float32).flatten()
            # Normalización L2 (el embed CLIP vive en la hiperesfera unitaria)
            clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            out[image_id] = clip_feat
        except Exception as exc:
            logger.warning(f"[{subject_id}] Error leyendo {pkl_file}: {exc}")

    logger.info(f"[{subject_id}] {len(out)} features CLIP cargados")
    return out


# ============================================================================
# ADAPTACIÓN DE DIMENSIÓN (PUENTE TEMPORAL 512 → 768)
# ============================================================================

_warned_about_shim = False


def adapt_embedding_to_unclip(
    feat_512: torch.Tensor,
    target_dim: int = UNCLIP_EMBED_DIM,
) -> torch.Tensor:
    """
    Lleva el embedding del dataset (ViT-B/32, 512-d) al espacio que SD unCLIP
    espera (ViT-L/14, 768-d).

    Estrategia: **zero-pad determinista**. La señal fMRI ocupa los primeros
    512 componentes; los 256 restantes son ceros. Tras la concatenación se
    renormaliza a L2=1 para mantener la convención de la hiperesfera CLIP.

    Esto NO es semánticamente correcto — los espacios ViT-B/32 y ViT-L/14
    están ROTADOS entre sí (son encoders independientes entrenados de
    forma contrastiva). El UNet de SD interpretará estos embeds como si
    fueran ViT-L/14, produciendo imágenes que obedecen a los primeros
    512 componentes de su manifold, no a la semántica cerebral.

    Se usa únicamente para validar la plumbing del pipeline. Reemplazar por
    el adapter fMRI→CLIP-ViT-L/14 (LoRA) cuando esté entrenado.

    Si ya viene en 768-d (post-adapter), esta función es pass-through.
    """
    global _warned_about_shim

    if feat_512.dim() == 1:
        feat_512 = feat_512.unsqueeze(0)  # (D,) → (1, D)

    current_dim = feat_512.shape[1]
    if current_dim == target_dim:
        return feat_512  # ya está en el espacio correcto

    if current_dim > target_dim:
        # Truncate si viniera más grande (no esperado en este proyecto).
        return feat_512[:, :target_dim]

    if not _warned_about_shim:
        logger.warning(
            "=" * 70 + "\n"
            f"SHIM ACTIVO: zero-pad {current_dim} → {target_dim} para alimentar SD unCLIP.\n"
            "Los resultados NO son el experimento final — son smoke test de plumbing.\n"
            "TODO: reemplazar por adapter fMRI→CLIP-ViT-L/14 (ver MIGRATION.md §7).\n"
            + "=" * 70
        )
        _warned_about_shim = True

    pad = torch.zeros(
        feat_512.shape[0],
        target_dim - current_dim,
        dtype=feat_512.dtype,
        device=feat_512.device,
    )
    padded = torch.cat([feat_512, pad], dim=1)
    # Renormalizar a la hiperesfera unitaria (convención CLIP)
    padded = padded / padded.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return padded


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_subject(
    pipeline,
    subject_id: str,
    output_dir: Path,
    num_inference_steps: int = INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    limit: int | None = None,
) -> int:
    """
    Procesa un sujeto completo. Devuelve número de imágenes guardadas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_clip_features_for_subject(subject_id)

    if not features:
        logger.warning(f"[{subject_id}] Sin features. Saltando.")
        return 0

    items = list(features.items())
    if limit is not None:
        items = items[:limit]
        logger.info(f"[{subject_id}] Limitado a {limit} imágenes")

    saved = 0
    t_start = time.perf_counter()

    for idx, (image_id, clip_feat) in enumerate(items, 1):
        out_path = output_dir / f"{subject_id}_{image_id}_sd_unclip.png"
        if out_path.exists():
            logger.info(f"[{subject_id}] ({idx}/{len(items)}) {image_id} — existe, salto")
            saved += 1
            continue

        brain_embed = adapt_embedding_to_unclip(clip_feat)  # (1, 768)

        try:
            current_prompt = SD_PRIOR_PROMPTS[idx % len(SD_PRIOR_PROMPTS)]
            img = reconstruct_from_embedding(
                pipeline,
                brain_embed,
                prompt=current_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=GLOBAL_SEED,  # mismo embed + mismo seed = mismo output
            )
            img.save(out_path)
            saved += 1
            logger.info(f"[{subject_id}] ({idx}/{len(items)}) {image_id} → {out_path.name}")
        except Exception as exc:
            logger.error(f"[{subject_id}] Fallo en {image_id}: {exc}")
            continue

    dt = time.perf_counter() - t_start
    per_img = dt / max(saved, 1)
    logger.info(f"[{subject_id}] {saved}/{len(items)} imágenes en {dt:.1f}s ({per_img:.1f}s/img)")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description="Fase 2 — inferencia SD 2.1 unCLIP desde fMRI→CLIP")
    ap.add_argument(
        "--subjects",
        nargs="+",
        default=config.PROCESSING_CONFIG["subjects"],
        help="Sujetos a procesar (default: todos en config.py)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Límite de imágenes por sujeto")
    ap.add_argument("--cpu", action="store_true", help="Forzar CPU")
    ap.add_argument("--steps", type=int, default=INFERENCE_STEPS, help="Pasos DPM-Solver")
    ap.add_argument("--guidance", type=float, default=GUIDANCE_SCALE, help="CFG scale")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # ------------------------------------------------------------------
    # SEMILLA GLOBAL — antes de cualquier import de peso random
    # ------------------------------------------------------------------
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info(f"Device: {device} | seed={GLOBAL_SEED} | steps={args.steps} | cfg={args.guidance}")

    # ------------------------------------------------------------------
    # PIPELINE + SWAP DE SCHEDULER
    # ------------------------------------------------------------------
    pipeline = load_sd_unclip_pipeline(device=device, seed=GLOBAL_SEED)

    # DPMSolverMultistepScheduler: solver multi-step ODE de orden 2.
    # Produce muestras comparables a DDIM/PNDM en 20-30 pasos en lugar
    # de 50+. Respeta la convención de `config` del scheduler anterior
    # (alpha_cumprod, prediction_type), así que el swap es seguro.
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    logger.info(f"Scheduler: {type(pipeline.scheduler).__name__}")

    # ------------------------------------------------------------------
    # LOOP POR SUJETOS
    # ------------------------------------------------------------------
    total_saved = 0
    for subject_id in args.subjects:
        logger.info("=" * 70)
        logger.info(f"SUJETO {subject_id}")
        logger.info("=" * 70)
        subject_out_dir = OUTPUT_ROOT / subject_id
        total_saved += run_subject(
            pipeline,
            subject_id,
            subject_out_dir,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            limit=args.limit,
        )

    logger.info("=" * 70)
    logger.info(f"Fase 2 completada — {total_saved} imágenes en {OUTPUT_ROOT}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
