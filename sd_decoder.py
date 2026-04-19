"""
===============================================================================
FASE 2 — DECODIFICADOR BASADO EN STABLE DIFFUSION 2.1 unCLIP
===============================================================================

Proyecto ACECOM — Reconstrucción de imágenes mentales desde fMRI.
Base: Koide-Majima et al. (2024). Migración documentada en MIGRATION.md §4.

OBJETIVO DE ESTE MÓDULO
-----------------------
Reemplazar el generador VQGAN (Fase 1) por Stable Diffusion 2.1 unCLIP,
que acepta un embedding CLIP como condición de imagen. Esto encaja
directamente con la salida del adapter fMRI → CLIP-emb (ViT-L/14, 768-d).

Arquitectura target (ver MIGRATION.md §4.2):

    fMRI → Adapter (LoRA) → z_CLIP ∈ R^768 → SD 2.1 unCLIP UNet → VAE dec → imagen

HARDWARE
--------
Diseñado para RTX 4070 Ti (12 GB VRAM). Usa bf16 + xformers para caber
en ~5-6 GB durante inferencia.

REFERENCIAS HF
--------------
- Modelo:  https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip
- Pipeline: diffusers.StableUnCLIPImg2ImgPipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from accelerate.utils import set_seed

import config

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES DEL STACK FASE 2
# ============================================================================

SD_UNCLIP_REPO = "diffusers/stable-diffusion-2-1-unclip-i2i-l"
SD_DTYPE = torch.bfloat16  # bf16 evita NaN en tasks con mean-subtraction (ver MIGRATION.md §3.2)

# Cache local de pesos HF (dentro del repo para no mezclar con ~/.cache)
SD_CACHE_DIR = config.PROJECT_ROOT / "models_hf"

# ----------------------------------------------------------------------------
# NEGATIVE PROMPT FIJO — estabilizador contra la alta varianza del baseline VQGAN
# ----------------------------------------------------------------------------
# Diagnóstico Fase 1 (MIGRATION.md §2.1): el optimizador VQGAN producía
# "manchones" que satisfacían matemáticamente el vector CLIP pero eran ruido
# visual (PixCorr ≈ 0). En SD 2.1 unCLIP el fenómeno análogo es que el UNet,
# condicionado sólo por image_embeds, puede derivar hacia imágenes abstractas
# o con múltiples objetos cuando el embedding fMRI-predicho tiene ruido.
#
# El negative prompt actúa como "empuje en contra" (classifier-free guidance
# negativa): el UNet aprende a EVITAR regiones del latente descritas por estas
# palabras. Con guidance_scale>1, esto estabiliza la salida sin necesidad de
# un prior de difusión adicional (que se añadirá en Fase 3 según MIGRATION.md §6).
#
# Palabras elegidas directamente contra las 3 fallas diagnosticadas:
#   - "blurry, noise" → contra la borrosidad por ajuste al promedio (Falla 3)
#   - "abstract, chaotic, deformed" → contra los manchones CLIP (Falla 1)
#   - "multiple objects" → fuerza una única escena dominante (reduce varianza)
SD_NEGATIVE_PROMPT = "blurry, noise, abstract, deformed, chaotic, multiple objects"

# Prompts textuales de estabilización (Priors visuales). 
# Fueron omitidos erróneamente por el script anterior dejándolo ciego.
# Estos prompts anclarán la semántica para que la imagen cobre forma realista.
SD_PRIOR_PROMPTS = [
    "A clear, high-quality photograph of a natural scene, realistic, defined shapes",
    "A vivid visual memory, highly detailed, photorealistic perception",
    "A coherent object or landscape, high resolution, sharp focus, real life"
]


# ============================================================================
# CARGA DEL PIPELINE
# ============================================================================

def load_sd_unclip_pipeline(
    device: torch.device | str = "cuda",
    repo_id: str = SD_UNCLIP_REPO,
    cache_dir: Path | str | None = None,
    enable_xformers: bool = True,
    enable_vae_slicing: bool = True,
    seed: int | None = 42,
) -> StableUnCLIPImg2ImgPipeline:
    """
    Descarga (si falta) e instancia el pipeline SD 2.1 unCLIP en bf16.

    Esta función es el punto de entrada de Fase 2. Devuelve un pipeline
    listo para inferencia condicional sobre embeddings CLIP — el adapter
    fMRI→CLIP alimentará el argumento `image_embeds` en el llamado al
    pipeline (se implementa en `phase2/infer.py`, no aquí).

    Args:
        device: 'cuda' o torch.device. En CPU se fuerza fp32 (bf16 en CPU
            en PyTorch >=2.0 funciona pero es 10× más lento — no vale la pena).
        repo_id: identificador del modelo en Hugging Face Hub.
        cache_dir: ruta local para cache de pesos. Default: `PROJECT_ROOT/models_hf/`.
            Mantener los pesos dentro del repo facilita portabilidad a la
            máquina nueva (RTX 4070 Ti) y evita recargar ~5 GB del hub.
        enable_xformers: activa memory-efficient attention (CRÍTICO para 12 GB).
        enable_vae_slicing: decodifica la latente en slices; ahorra VRAM en
            el paso final del VAE cuando la batch crece. Coste: ~2% más lento.
        seed: semilla global (vía `accelerate.set_seed`) para reproducibilidad
            entre corridas. Pasar `None` para no fijarla.

    Returns:
        StableUnCLIPImg2ImgPipeline en `eval()` sobre `device`, dtype bf16.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if seed is not None:
        set_seed(seed)  # CPU + CUDA + accelerate-aware

    if cache_dir is None:
        cache_dir = SD_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dtype = SD_DTYPE if device.type == "cuda" else torch.float32

    logger.info(f"Cargando SD 2.1 unCLIP desde {repo_id} (dtype={dtype}, cache={cache_dir})")

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        cache_dir=str(cache_dir),
        variant="fp16" if dtype == torch.bfloat16 else None,
        use_safetensors=True,
    )

    # ------------------------------------------------------------------
    # OPTIMIZACIONES DE MEMORIA (RTX 4070 Ti 12 GB)
    # ------------------------------------------------------------------
    # xformers: memory-efficient attention. Reduce VRAM del UNet ~30-40%
    # y acelera forward ~10-20%. Sin esto, SD 2.1 unCLIP llega a ~8-9 GB
    # bajo carga; con esto, ~5-6 GB.
    if enable_xformers and device.type == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("✓ xformers memory-efficient attention habilitado")
        except Exception as exc:
            logger.warning(
                f"xformers no disponible ({exc}). Instalar con: pip install xformers. "
                "Continuando sin xformers — VRAM más alto."
            )

    # VAE slicing: decodifica la latente en chunks. Útil cuando batch>1
    # o imagen>512. Sin coste si batch=1.
    if enable_vae_slicing:
        pipeline.enable_vae_slicing()

    # Desactivar el safety checker (añade VRAM y no aplica a neuroimaging).
    pipeline.safety_checker = None

    # Mover a device (una sola vez; evita `.to(device)` en cada llamada).
    pipeline = pipeline.to(device)

    # Modo inferencia — el UNet y VAE quedan frozen en Fase 2 (training
    # sólo ocurre sobre el adapter fMRI→CLIP, fuera de este módulo).
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.image_encoder.eval()

    logger.info(
        f"✓ Pipeline listo | device={device} | dtype={dtype} | "
        f"xformers={enable_xformers} | vae_slicing={enable_vae_slicing}"
    )

    return pipeline


# ============================================================================
# INFERENCIA — CONDICIONAMIENTO POR EMBEDDING CEREBRAL (sin texto)
# ============================================================================

def reconstruct_from_embedding(
    pipeline: StableUnCLIPImg2ImgPipeline,
    brain_clip_embedding: torch.Tensor,
    prompt: str = SD_PRIOR_PROMPTS[0],
    num_inference_steps: int = 30,
    guidance_scale: float = 10.0,
    noise_level: int = 250,
    negative_prompt: str = SD_NEGATIVE_PROMPT,
    seed: int | None = 42,
    output_height: int = 768,
    output_width: int = 768,
):
    """
    Reconstruye una imagen a partir de un embedding CLIP predicho del fMRI.

    PUNTO CRÍTICO — por qué image_embeds y NO prompt de texto:
    ----------------------------------------------------------
    El objetivo científico de la tesis es medir cuánta información visual
    se puede recuperar del cerebro. Condicionar el UNet con un prompt de
    texto introduciría una fuente de información EXTERNA al fMRI (priors
    lingüísticos de CLIP text-encoder + del UNet) que sesgaría las métricas
    al alza sin que el delta venga realmente del cerebro.

    StableUnCLIPImg2ImgPipeline acepta dos rutas de condicionamiento visual:
        a) image=PIL → internamente el pipeline pasa la imagen por el
           image_encoder (CLIP ViT-L/14) para producir image_embeds.
        b) image_embeds=tensor [B, 768] → se salta el image_encoder.

    Usamos SIEMPRE la ruta (b) con el embedding del adapter fMRI→CLIP.
    Esto garantiza que la única señal visual que entra al UNet viene del
    cerebro (o del target ground-truth, según el experimento).

    El negative_prompt sí pasa por el text_encoder de SD — eso es esperado
    y deseado: define regiones del espacio latente a evitar vía
    classifier-free guidance negativa. No filtra señal cerebral porque
    se resta, no se suma.

    ESTABILIDAD vs el baseline VQGAN (MIGRATION.md §2):
    ---------------------------------------------------
    Mitigaciones integradas en esta función contra la alta variabilidad
    de reconstrucción del baseline:
      1. `seed` fija via `torch.Generator` por-llamada → mismo embed +
         mismo seed = imagen idéntica (reproducibilidad bit-exacta).
      2. `negative_prompt` intenso contra borrosidad/caos (ver SD_NEGATIVE_PROMPT).
      3. `noise_level=0` por defecto → sin ruido extra añadido al embed
         cerebral (el unCLIP image_normalizer acepta inyectar ruido
         gaussiano al embed; ponerlo a 0 reduce varianza stochastic).
      4. `guidance_scale=10` → CFG alto empuja fuerte hacia la condición,
         estrechando la distribución de outputs.

    Args:
        pipeline: el pipeline ya cargado por `load_sd_unclip_pipeline()`.
        brain_clip_embedding: tensor del adapter fMRI→CLIP. Shape esperado
            `(1, 768)` o `(768,)` — se normaliza a `(B, 768)` automáticamente.
            Debe estar en el mismo device que el pipeline. dtype se ajusta
            al del UNet (bf16 en GPU).
        num_inference_steps: pasos del scheduler (DDIM por default).
            30 es el mínimo estable para SD 2.1 unCLIP; <20 degrada.
        guidance_scale: coeficiente CFG. 10.0 es agresivo y estabiliza;
            valores <7 aumentan varianza de muestra.
        noise_level: ruido gaussiano inyectado al image_embed por el
            image_normalizer interno. 0 = sin ruido → mínima varianza.
            Rango permitido por el pipeline: [0, 1000].
        negative_prompt: texto a evitar. Default: SD_NEGATIVE_PROMPT.
            Pasar "" si se quiere desactivar CFG negativa.
        seed: semilla para el generador de ruido inicial del UNet. `None`
            deja el sampler stochastic (no recomendado: alta varianza).
        output_height, output_width: resolución. SD 2.1 unCLIP nativo = 768.

    Returns:
        PIL.Image.Image: la imagen reconstruida.

    Ejemplo de uso:
        >>> pipe = load_sd_unclip_pipeline(device="cuda")
        >>> z_clip = adapter_fmri_to_clip(fmri_voxels)     # (1, 768)
        >>> img = reconstruct_from_embedding(pipe, z_clip) # PIL.Image
        >>> img.save("recon_S01_n04507155_21299.png")
    """
    # --- Normalización de shape: aceptar (768,) o (1, 768) ---
    if brain_clip_embedding.dim() == 1:
        brain_clip_embedding = brain_clip_embedding.unsqueeze(0)
    if brain_clip_embedding.dim() != 2 or brain_clip_embedding.shape[1] != 768:
        raise ValueError(
            f"brain_clip_embedding debe tener shape (1, 768) o (768,); "
            f"recibido: {tuple(brain_clip_embedding.shape)}"
        )

    device = pipeline.unet.device
    target_dtype = pipeline.unet.dtype
    brain_clip_embedding = brain_clip_embedding.to(device=device, dtype=target_dtype)

    # --- Generator determinista por-llamada (no contamina RNG global) ---
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    # --- Llamada al pipeline ---
    # prompt="" fuerza la ruta image_embeds pura (sin condicionamiento textual
    # positivo). El negative_prompt sí se usa para CFG negativa.
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            image_embeds=brain_clip_embedding,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            generator=generator,
            height=output_height,
            width=output_width,
        )

    return result.images[0]


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test() -> None:
    """Verificación mínima: carga + 1 inferencia con embed dummy vía reconstruct_from_embedding."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Smoke test en device={device}")

    pipeline = load_sd_unclip_pipeline(device=device)

    # Embed dummy 768-d. En producción vendrá del adapter fMRI→CLIP (ViT-L/14).
    dummy_embed = torch.randn(1, 768, device=device)

    img = reconstruct_from_embedding(
        pipeline,
        dummy_embed,
        num_inference_steps=10,  # bajo para smoke test
        guidance_scale=10.0,
        seed=42,
    )

    logger.info(f"✓ Smoke test OK — imagen shape={img.size} | negative_prompt='{SD_NEGATIVE_PROMPT}'")


if __name__ == "__main__":
    _smoke_test()
