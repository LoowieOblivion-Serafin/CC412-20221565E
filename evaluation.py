"""
===============================================================================
MÓDULO DE EVALUACIÓN CUANTITATIVA DE RECONSTRUCCIONES
===============================================================================

Calcula las métricas estándar del campo sobre las reconstrucciones generadas
por `main_local_decoder.py`, contrastándolas contra los estímulos originales
almacenados en `mental_img_recon-main/ref_images/target_images.pkl`.

Métricas producidas:
    - SSIM           : similitud estructural (bajo nivel)
    - PixCorr        : correlación de Pearson sobre píxeles
    - PSNR           : relación señal/ruido (dB)
    - LPIPS          : distancia perceptual aprendida (si está instalado)
    - CLIP cosine    : similitud semántica en espacio CLIP
    - Pairwise ID    : accuracy de identificación 2-vías (métrica clave del
                       paper Koide-Majima et al. 2024)

Uso:
    py -3.12 evaluation.py                 # todos los sujetos del config
    py -3.12 evaluation.py --subjects S01  # sujetos específicos
    py -3.12 evaluation.py --no-html       # solo CSV

Salida:
    output_reconstructions/
        metrics_<subject>.csv
        metrics_summary.csv
        report_metrics.html

Referencias metodológicas:
    Koide-Majima & Nishimoto (2024) Neural Networks 170:349-363
    Scotti et al. (NeurIPS 2023) MindEye — pairwise identification
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluation")

# ---------------------------------------------------------------------------
# Dependencias opcionales (degradación elegante si falta alguna)
# ---------------------------------------------------------------------------
try:
    from skimage.metrics import structural_similarity as _ssim
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    log.warning("scikit-image no instalado; SSIM/PSNR se omitirán.")

try:
    from scipy.stats import pearsonr as _pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    log.warning("scipy no instalado; PixCorr se omitirá.")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.error("PyTorch no instalado. No se podrá calcular CLIP ni LPIPS.")

try:
    import lpips as _lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    log.info("lpips no instalado; métrica LPIPS se omitirá. "
             "Instalar con: pip install lpips")

HAS_CLIP = False
if HAS_TORCH:
    try:
        sys.path.insert(0, str(config.REPOS["clip"]))
        import clip as _clip
        HAS_CLIP = True
    except Exception as err:  # noqa: BLE001
        log.warning("CLIP no disponible (%s); similitud semántica omitida.", err)


# ---------------------------------------------------------------------------
# Datos: carga de estímulos y reconstrucciones
# ---------------------------------------------------------------------------
IMG_SIZE = config.PROCESSING_CONFIG["image_size"]  # 256 por defecto
TARGET_PKL = config.REPOS["mental_img_recon"] / "ref_images" / "target_images.pkl"


@dataclass
class StimulusPair:
    """Par alineado estímulo–reconstrucción para una etiqueta."""
    label: str
    subject: str
    gt: np.ndarray          # uint8 HxWx3
    recon: np.ndarray       # uint8 HxWx3


def load_stimuli() -> Dict[str, np.ndarray]:
    """Lee target_images.pkl y devuelve {label: ndarray uint8 HxWx3}."""
    if not TARGET_PKL.exists():
        raise FileNotFoundError(
            f"No se encontró {TARGET_PKL}. Verifica que mental_img_recon-main "
            f"esté clonado."
        )
    with open(TARGET_PKL, "rb") as f:
        data = pickle.load(f)
    tile = data["target_images"]
    h, w = data["imsize"]
    positions: Dict[str, Tuple[int, int]] = data["target_positions"]
    stimuli: Dict[str, np.ndarray] = {}
    for label, (y, x) in positions.items():
        crop = tile[y:y + h, x:x + w]
        stimuli[label] = np.ascontiguousarray(crop)
    log.info("Cargados %d estímulos originales desde target_images.pkl", len(stimuli))
    return stimuli


def parse_reconstruction_filename(filename: str, subject: str) -> Optional[str]:
    """Extrae el label de `<subject>_<label>_reconstructed.png`."""
    prefix = f"{subject}_"
    suffix = "_reconstructed.png"
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        return None
    return filename[len(prefix):-len(suffix)]


def load_reconstructions(subject: str) -> Dict[str, np.ndarray]:
    """Lee todas las reconstrucciones PNG de un sujeto."""
    subject_dir = config.DATA_DIRS["output"] / subject
    if not subject_dir.is_dir():
        log.warning("Directorio no encontrado: %s", subject_dir)
        return {}
    recons: Dict[str, np.ndarray] = {}
    for png in sorted(subject_dir.glob("*_reconstructed.png")):
        label = parse_reconstruction_filename(png.name, subject)
        if label is None:
            continue
        img = np.asarray(Image.open(png).convert("RGB"))
        recons[label] = img
    log.info("Sujeto %s: %d reconstrucciones cargadas", subject, len(recons))
    return recons


def align_pairs(subject: str,
                stimuli: Dict[str, np.ndarray],
                recons: Dict[str, np.ndarray]) -> List[StimulusPair]:
    """Empareja reconstrucciones con su ground-truth redimensionado a IMG_SIZE."""
    pairs: List[StimulusPair] = []
    skipped: List[str] = []
    for label, recon in recons.items():
        if label not in stimuli:
            skipped.append(label)
            continue
        gt = stimuli[label]
        if gt.shape[:2] != (IMG_SIZE, IMG_SIZE):
            gt = np.asarray(
                Image.fromarray(gt).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            )
        if recon.shape[:2] != (IMG_SIZE, IMG_SIZE):
            recon = np.asarray(
                Image.fromarray(recon).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            )
        pairs.append(StimulusPair(label=label, subject=subject, gt=gt, recon=recon))
    if skipped:
        log.info(
            "Sujeto %s: %d estímulos sin ground-truth (p.ej. fixation): %s",
            subject, len(skipped), ", ".join(skipped)
        )
    return pairs


# ---------------------------------------------------------------------------
# Modelos perceptuales cacheados
# ---------------------------------------------------------------------------
@dataclass
class PerceptualModels:
    device: str
    lpips_fn: Optional[object] = None
    clip_model: Optional[object] = None
    clip_preprocess: Optional[object] = None


def build_models(device: Optional[str] = None) -> PerceptualModels:
    if not HAS_TORCH:
        return PerceptualModels(device="cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = PerceptualModels(device=device)
    if HAS_LPIPS:
        log.info("Cargando LPIPS (AlexNet) en %s", device)
        models.lpips_fn = _lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()
    if HAS_CLIP:
        variant = config.MODEL_CONFIG["clip_variant"]
        log.info("Cargando CLIP %s en %s", variant, device)
        clip_model, preprocess = _clip.load(variant, device=device, jit=False)
        clip_model.eval()
        models.clip_model = clip_model
        models.clip_preprocess = preprocess
    return models


# ---------------------------------------------------------------------------
# Métricas individuales
# ---------------------------------------------------------------------------
def pixel_metrics(recon: np.ndarray, gt: np.ndarray) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"ssim": None, "pixcorr": None, "psnr": None}
    if HAS_SKIMAGE:
        out["ssim"] = float(_ssim(gt, recon, channel_axis=2, data_range=255))
        # PSNR=inf cuando MSE=0 (imágenes idénticas); reemplazamos por NaN para CSV.
        psnr_val = float(_psnr(gt, recon, data_range=255))
        out["psnr"] = psnr_val if np.isfinite(psnr_val) else float("nan")
    if HAS_SCIPY:
        r, _ = _pearsonr(gt.flatten().astype(np.float64),
                         recon.flatten().astype(np.float64))
        out["pixcorr"] = float(r)
    return out


def _to_lpips_tensor(img: np.ndarray, device: str) -> "torch.Tensor":
    arr = np.ascontiguousarray(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0  # [-1, 1]
    return t.unsqueeze(0).to(device)


def lpips_metric(pair: StimulusPair, models: PerceptualModels) -> Optional[float]:
    if models.lpips_fn is None:
        return None
    with torch.no_grad():
        d = models.lpips_fn(
            _to_lpips_tensor(pair.recon, models.device),
            _to_lpips_tensor(pair.gt, models.device),
        )
    return float(d.item())


def _clip_embed(imgs: List[np.ndarray], models: PerceptualModels) -> "torch.Tensor":
    assert models.clip_model is not None and models.clip_preprocess is not None
    tensors = [models.clip_preprocess(Image.fromarray(im)) for im in imgs]
    batch = torch.stack(tensors).to(models.device)
    with torch.no_grad():
        feats = models.clip_model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return feats


def clip_similarity(pairs: List[StimulusPair],
                    models: PerceptualModels) -> Dict[str, float]:
    """Devuelve {label: cosine(recon, gt)} en espacio CLIP-ViT-B/32."""
    if models.clip_model is None or not pairs:
        return {p.label: float("nan") for p in pairs}
    recons_feat = _clip_embed([p.recon for p in pairs], models)
    gts_feat = _clip_embed([p.gt for p in pairs], models)
    cos = (recons_feat * gts_feat).sum(dim=-1).detach().cpu().numpy()
    return {pair.label: float(c) for pair, c in zip(pairs, cos)}


def pairwise_identification(pairs: List[StimulusPair],
                            models: PerceptualModels) -> Dict[str, float]:
    """
    Pairwise 2-way identification accuracy (Koide-Majima §3.4, Scotti 2023).

    Para cada par i, y para cada otro par j != i:
        correcto si sim(recon_i, gt_i) > sim(recon_i, gt_j)

    accuracy = #(correcto) / #(pares (i,j) con i != j).
    Chance = 0.5. Koide-Majima reporta 75.6% para imagery.
    """
    n = len(pairs)
    if n < 2 or models.clip_model is None:
        return {"n": n, "pairwise_acc": float("nan")}
    recons_feat = _clip_embed([p.recon for p in pairs], models)
    gts_feat = _clip_embed([p.gt for p in pairs], models)
    sim = (recons_feat @ gts_feat.T).detach().cpu().numpy()  # [n, n]
    diag = np.diag(sim)
    # Conteo de j != i donde sim(i,i) > sim(i,j).
    mask = np.ones_like(sim, dtype=bool)
    np.fill_diagonal(mask, False)
    wins = (diag[:, None] > sim).astype(np.float64)
    wins = wins[mask].reshape(n, n - 1)
    acc = float(wins.mean())
    return {"n": n, "pairwise_acc": acc}


# ---------------------------------------------------------------------------
# Orquestación por sujeto
# ---------------------------------------------------------------------------
@dataclass
class PerImageRow:
    subject: str
    label: str
    ssim: Optional[float] = None
    pixcorr: Optional[float] = None
    psnr: Optional[float] = None
    lpips: Optional[float] = None
    clip_sim: Optional[float] = None


def evaluate_subject(subject: str,
                     stimuli: Dict[str, np.ndarray],
                     models: PerceptualModels) -> Tuple[List[PerImageRow], Dict[str, float]]:
    recons = load_reconstructions(subject)
    pairs = align_pairs(subject, stimuli, recons)
    rows: List[PerImageRow] = []
    clip_sims = clip_similarity(pairs, models)
    for p in pairs:
        px = pixel_metrics(p.recon, p.gt)
        row = PerImageRow(
            subject=subject,
            label=p.label,
            ssim=px["ssim"],
            pixcorr=px["pixcorr"],
            psnr=px["psnr"],
            lpips=lpips_metric(p, models),
            clip_sim=clip_sims.get(p.label),
        )
        rows.append(row)
    summary = {
        "subject": subject,
        "n_pairs": len(pairs),
        **_aggregate(rows),
        **pairwise_identification(pairs, models),
    }
    return rows, summary


def _aggregate(rows: List[PerImageRow]) -> Dict[str, float]:
    keys = ["ssim", "pixcorr", "psnr", "lpips", "clip_sim"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [getattr(r, k) for r in rows if getattr(r, k) is not None and np.isfinite(getattr(r, k))]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# Exportación de resultados
# ---------------------------------------------------------------------------
def write_csv(rows: List[PerImageRow], path: Path) -> None:
    if not rows:
        return
    header = list(asdict(rows[0]).keys())
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(
            "" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))
            for v in asdict(r).values()
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Escrito %s (%d filas)", path, len(rows))


def write_summary_csv(summaries: List[Dict[str, float]], path: Path) -> None:
    if not summaries:
        return
    header = list(summaries[0].keys())
    lines = [",".join(header)]
    for s in summaries:
        lines.append(",".join(
            f"{v:.6f}" if isinstance(v, float) and np.isfinite(v)
            else "" if v is None
            else str(v)
            for v in s.values()
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Escrito %s", path)


def write_html_report(rows: List[PerImageRow],
                      summaries: List[Dict[str, float]],
                      output_dir: Path,
                      stimuli: Dict[str, np.ndarray]) -> None:
    """HTML con tabla resumen y grid lado a lado (GT | recon) por imagen."""
    gt_cache = output_dir / "_gt_cache"
    gt_cache.mkdir(exist_ok=True)
    for label, arr in stimuli.items():
        out_png = gt_cache / f"{label}.png"
        if not out_png.exists():
            Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC).save(out_png)

    # Tabla resumen.
    summary_header = ["subject", "n_pairs", "mean_ssim", "mean_pixcorr",
                      "mean_psnr", "mean_lpips", "mean_clip_sim",
                      "pairwise_acc", "n"]
    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    summary_rows_html = "".join(
        "<tr>" + "".join(f"<td>{cell(s.get(k))}</td>" for k in summary_header) + "</tr>"
        for s in summaries
    )

    # Grid por sujeto.
    subjects_html = []
    by_subject: Dict[str, List[PerImageRow]] = {}
    for r in rows:
        by_subject.setdefault(r.subject, []).append(r)
    for subject, sub_rows in by_subject.items():
        cards = []
        for r in sub_rows:
            gt_src = f"_gt_cache/{r.label}.png"
            recon_src = f"{subject}/{subject}_{r.label}_reconstructed.png"
            metrics_line = (
                f"SSIM {cell(r.ssim)} · PixCorr {cell(r.pixcorr)} · "
                f"LPIPS {cell(r.lpips)} · CLIP {cell(r.clip_sim)}"
            )
            cards.append(f"""
            <div class="card">
                <div class="pair">
                    <figure><img src="{gt_src}"><figcaption>GT</figcaption></figure>
                    <figure><img src="{recon_src}"><figcaption>Recon</figcaption></figure>
                </div>
                <p class="label">{r.label}</p>
                <p class="metrics">{metrics_line}</p>
            </div>
            """)
        subjects_html.append(
            f"<section><h2>{subject}</h2><div class='grid'>"
            + "".join(cards) + "</div></section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Métricas de Reconstrucción — ACECOM</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 24px; background:#fafafa; color:#222; }}
 h1 {{ margin-bottom: 4px; }}
 .subtitle {{ color:#666; margin-top:0; }}
 table {{ border-collapse: collapse; margin: 16px 0 32px; background:white;
         box-shadow:0 1px 3px rgba(0,0,0,.1); }}
 th, td {{ padding: 8px 14px; border-bottom:1px solid #eee; text-align:right;
          font-variant-numeric: tabular-nums; }}
 th:first-child, td:first-child {{ text-align:left; }}
 th {{ background:#2c3e50; color:white; }}
 .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
         gap: 16px; }}
 .card {{ background:white; border-radius:6px; padding:10px;
         box-shadow:0 1px 2px rgba(0,0,0,.08); }}
 .pair {{ display:flex; gap:6px; }}
 .pair figure {{ margin:0; flex:1; text-align:center; }}
 .pair img {{ width:100%; border-radius:4px; }}
 .pair figcaption {{ font-size:11px; color:#666; }}
 .label {{ font-size:12px; font-family: monospace; color:#444; margin:6px 0 2px; }}
 .metrics {{ font-size:11px; color:#666; margin:0; }}
</style></head><body>
<h1>Métricas de Reconstrucción</h1>
<p class="subtitle">Baseline Koide-Majima et al. (2024) · VGG19 + CLIP + VQGAN</p>

<h2>Resumen por sujeto</h2>
<table>
 <thead><tr>{''.join(f'<th>{h}</th>' for h in summary_header)}</tr></thead>
 <tbody>{summary_rows_html}</tbody>
</table>

{''.join(subjects_html)}

<footer style="color:#888;font-size:11px;margin-top:32px;">
 Generado por <code>evaluation.py</code>.
 Métricas: SSIM/PSNR (skimage), PixCorr (scipy), LPIPS (AlexNet), CLIP ViT-B/32.
</footer>
</body></html>
"""
    (output_dir / "report_metrics.html").write_text(html, encoding="utf-8")
    log.info("Escrito %s", output_dir / "report_metrics.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(subjects: Iterable[str], write_html: bool = True) -> None:
    output_dir = config.DATA_DIRS["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    stimuli = load_stimuli()
    models = build_models()

    all_rows: List[PerImageRow] = []
    summaries: List[Dict[str, float]] = []

    for subject in subjects:
        rows, summary = evaluate_subject(subject, stimuli, models)
        if not rows:
            log.warning("Sujeto %s: sin pares evaluables, se omite.", subject)
            continue
        all_rows.extend(rows)
        summaries.append(summary)
        write_csv(rows, output_dir / f"metrics_{subject}.csv")
        log.info(
            "[%s] n=%d · SSIM=%.3f · PixCorr=%.3f · CLIP=%.3f · pairwise=%.3f",
            subject, summary["n_pairs"],
            summary["mean_ssim"], summary["mean_pixcorr"],
            summary["mean_clip_sim"], summary["pairwise_acc"],
        )

    if summaries:
        write_summary_csv(summaries, output_dir / "metrics_summary.csv")
        if write_html:
            write_html_report(all_rows, summaries, output_dir, stimuli)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=config.PROCESSING_CONFIG["subjects"],
        help="Lista de sujetos (por defecto los de config.py).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="No generar reporte HTML (solo CSV).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.subjects, write_html=not args.no_html)
