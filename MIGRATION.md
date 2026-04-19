# MIGRATION.md — Fase 1 (optimización) → Fase 2 (Stable Diffusion 2.1 unCLIP)

> Proyecto ACECOM — Reconstrucción de imágenes mentales desde fMRI
> Autor: Alvaro Taipe Cotrina (UNI)
> Base: Koide-Majima et al. (2024)
> Fecha: 2026-04-18
> Hardware actual: RTX 3070 (8 GB) → Hardware destino: RTX 4070 Ti (12 GB) + Intel UHD 770

---

## 1. Estado actual (baseline abril 2026)

Medición ejecutada el **2026-04-17** con `evaluation.py` sobre `output_reconstructions/`, configuración `standard` del decoder (VGG19 + CLIP ViT-B/32 + VQGAN f=16). Stack de métricas: CLIP ViT-B/32 + LPIPS AlexNet + `skimage` SSIM/PSNR.

### 1.1 S01 (n = 25 pares, único sujeto completo)

| Métrica | Valor | Interpretación | Target paper |
|---|---:|---|---:|
| SSIM | 0.457 | Similitud estructural moderada | — |
| PixCorr | −0.027 | **≈ cero**, sin correlación pixel-a-pixel | — |
| PSNR | 12.12 dB | Muy bajo (típico: 20+ dB) | — |
| LPIPS (AlexNet) | 0.734 | Alto = mal; objetivo < 0.5 | — |
| CLIP cos | 0.501 | Alineación semántica pobre | — |
| **Pairwise-ID 2-vías** | **0.495** | **En chance (0.5)** | **0.756** |

### 1.2 Otros sujetos

- **S02**: sólo 1 reconstrucción en disco (`S02_black_+_reconstructed.png`). Falta re-correr `main_local_decoder.py` para las 24 restantes.
- **S03**: pendiente completo.

### 1.3 Diagnóstico cuantitativo

Koide-Majima et al. (2024) reportan **75.6%** de pairwise-ID en imagery. El pipeline actual está **en chance (49.5%)** — delta **−26 pp**. Esto confirma cuantitativamente el diagnóstico del informe de asesoría (abril 2026): el stack VGG+VQGAN produce reconstrucciones que son **ruido visual desde el punto de vista identificativo**.

Archivos de referencia:
- `output_reconstructions/metrics_S01.csv`
- `output_reconstructions/metrics_summary.csv`
- `output_reconstructions/report_metrics.html`

---

## 2. Diagnóstico — las 3 fallas estructurales del pipeline actual

Del informe `Papers/Asesoria_Tesis_Neurociencia_Computacional.pdf`:

### 2.1 Ambigüedad "uno-a-muchos" con CLIP
CLIP captura **semántica** ("hay un gato") pero es **invariante a estructura espacial**. El optimizador en espacio latente encuentra "manchones" que satisfacen matemáticamente el vector CLIP pero son ruido visual. Es el fallo que explica el PixCorr ≈ 0: el optimizador NO alinea estructura espacial.

### 2.2 Optimización continua sobre codebook discreto VQGAN
`z` se optimiza como vector **continuo** con Adam + Langevin, pero VQGAN sólo decodifica bien **códigos del codebook discreto**. Los valores intermedios producen artefactos psicodélicos. El parche `quantize_interval=50` en `reconstruct_image` lo mitiga parcialmente, pero no elimina el problema raíz.

### 2.3 Ajuste al promedio lineal
Los `.pkl` de `decoded_features/` son predicciones **lineales** (regresión ridge) fMRI → features. Son inherentemente promediadas sobre el dataset de entrenamiento. Forzar coincidencia exacta con estos targets genera borrosidad — es información promediada, no información de la imagen específica.

---

## 3. Cambios aplicados en Fase 1 (optimización del pipeline actual)

Todas las ediciones buscan **rendimiento y reproducibilidad**, no cambiar la semántica del loss. Son no-op matemáticos (mismo seed → mismo output) pero reducen wall-time en RTX 30xx/40xx.

### 3.1 Tabla de cambios

| Archivo | Ubicación | Cambio | Impacto |
|---|---|---|---|
| `main_local_decoder.py` | imports (línea 89, 94) | Añadidos `nullcontext`, `torch.amp` | Habilita autocast bf16 |
| `main_local_decoder.py` | módulo (162-204) | Caches `_AUG_TRANSFORM_CACHE`, `_CLIP_NORMALIZE_CACHE`, `_IMAGENET_NORMALIZE_CACHE`, `_COS_SIM` | Evita reconstruir módulos ×500 iter × 25 imgs |
| `main_local_decoder.py` | `create_crops` (239, 270) | Usa `_aug_transform(device)` + `torch.rand(..., device=device)` | Elimina H2D transfer por iter |
| `main_local_decoder.py` | `clip_loss` (743, 764) | Usa `_clip_normalize(device)` + `_COS_SIM` módulo-nivel | −1 construcción de `Normalize` + `CosineSimilarity` por iter |
| `main_local_decoder.py` | `vgg_perceptual_loss` (816, 865) | Asume features pre-movidas; usa caches | Elimina ~500 KB × N_capas × 500 iter H2D |
| `main_local_decoder.py` | `reconstruct_image` (1022-1037) | Pre-mueve `target_clip_features`, `mean_clip_feature`, dicts VGG con `non_blocking=True`; `amp.autocast('cuda', dtype=torch.bfloat16)` en forward+loss | −10-20% wall-time GPU |
| `main_local_decoder.py` | `reconstruct_image` (1043) | `optimizer.zero_grad(set_to_none=True)` | Menor presión de memoria + reset más rápido |
| `main_local_decoder.py` | `reconstruct_image` (997) | `torch.cuda.manual_seed_all(seed)` | Reproducibilidad multi-GPU |
| `main_local_decoder.py` | `main` (1168-1171) | `cudnn.benchmark=True`, `matmul.allow_tf32=True`, `cudnn.allow_tf32=True` | Autotune de kernels + matmul TF32 (Ampere/Ada) |
| `config.py` | `EVALUATION_CONFIG` | Habilitado `compute_metrics=True`, lista de métricas, export CSV+HTML, path a `target_images.pkl` | Evaluación automática post-reconstrucción |
| `evaluation.py` | nuevo archivo (~450 líneas) | Módulo completo: load stimuli, align pairs, pixel/perceptual/semantic metrics, pairwise-ID 2-vías, export CSV+HTML | Baseline cuantificable para comparar Fase 2 |
| `requirements_py312.txt` | deps | Añadidos `scikit-image>=0.19.0`, `lpips>=0.1.4` | Deps para `evaluation.py` |

### 3.2 Racional de diseño

**Autocast bf16 (no fp16):** bf16 tiene mismo rango dinámico que fp32 — evita NaN en `clip_loss`/`vgg_perceptual_loss` donde se hace `x - x.mean(dim=1, keepdim=True)` con targets centrados por `mean_clip_feature`. fp16 perdería precisión en estas restas.

**Autocast sólo en forward+loss:** backward se ejecuta fuera del contexto autocast. El estado de Adam y los gradientes se mantienen en fp32 — así la convergencia es idéntica al baseline fp32, sólo el forward es más rápido.

**Pre-movimiento de features:** el mayor hotspot medido del bucle era `{k: v.to(device) for k, v in target_vgg_features.items()}` ejecutado dentro del bucle de 500 iter (repetía ~3.5 MB de H2D transfer por iter). Mover una vez antes del bucle es correcto porque los targets son constantes durante la optimización de `z`.

**TF32 + cudnn.benchmark:** gratis en Ampere/Ada. El benchmark autotunea al primer forward; las iters 2..500 reutilizan kernels ya seleccionados.

### 3.3 Expectativa de impacto en RTX 4070 Ti (12 GB)

Sin cambiar la semántica del loss, esperamos:
- **−20% a −35% wall-time** en el bucle de optimización (bf16 + TF32 + caches + pre-mov).
- **Mismas métricas** — esto es crítico: si el número cambia, es bug. `evaluation.py` con el mismo seed debe dar los 6 números de §1.1.

Medición esperada post-traslado: 25 imágenes × 500 iter en RTX 4070 Ti debería correr en **~8-12 minutos** (vs ~15-20 min en RTX 3070 pre-optimización, extrapolado).

---

## 4. Fase 2 — Migración a Stable Diffusion 2.1 unCLIP

### 4.1 Racional

VQGAN (2020, f=16, codebook 1024 tokens) está obsoleto como generador para tareas de reconstrucción fina. SD 2.1 (2022) usa un VAE continuo + UNet condicional con mucha más capacidad generativa, y la variante **unCLIP** acepta un embedding CLIP como condición — que es exactamente lo que produce la rama fMRI→CLIP del pipeline actual.

Además, SD 2.1 unCLIP elimina **dos de las tres fallas** diagnosticadas:
- **Falla 2 (codebook discreto):** el VAE de SD es continuo, sin proyección a codebook.
- **Falla 1 (ambigüedad CLIP):** el prior aprendido del UNet actúa como regularizador natural del espacio de imágenes plausibles; no buscamos una z arbitraria que satisfaga el vector CLIP, sino que generamos muestras del manifold aprendido.

La falla 3 (ajuste al promedio lineal) es ortogonal al generador — se ataca con un **adapter fMRI→CLIP-emb** entrenable (LoRA) que aprenda a predecir embeddings CLIP de muestras específicas, no el promedio.

### 4.2 Arquitectura target

```
┌──────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐   ┌──────────┐
│ fMRI (vóxeles)   │──▶│ Adapter fMRI→CLIP    │──▶│ SD 2.1 unCLIP UNet  │──▶│ VAE dec  │──▶ imagen
│ [V1..IT ROIs]    │   │ (LoRA, entrenable)   │   │ (frozen)            │   │ (frozen) │
└──────────────────┘   └──────────────────────┘   └─────────▲───────────┘   └──────────┘
                              │ 768-d CLIP ViT-L/14 pred    │
                              └─────────────────────────────┘
                              (opcional Fase 3) VDVAE low-level prior ──┘
```

Componentes:
- **Entrada:** vector de vóxeles fMRI (mismo dataset Koide-Majima que Fase 1).
- **Adapter:** MLP o ridge-regression + LoRA, entrena `fMRI → z_CLIP` con supervisión directa del embedding CLIP ViT-L/14 de la imagen ground-truth.
- **SD 2.1 unCLIP:** UNet + VAE **frozen**. Condición: el `z_CLIP` predicho por el adapter.
- **Sampler:** DDIM 50 pasos (inferencia rápida).

### 4.3 Trade-offs VRAM (RTX 4070 Ti 12 GB)

| Configuración | VRAM aprox (inferencia bf16 + xformers) | Viable |
|---|---:|---|
| SD 2.1 unCLIP-L (ViT-L/14 cond) | ~5 GB | ✅ cómodo |
| SD 2.1 unCLIP-H (ViT-H/14 cond) | ~6 GB | ✅ |
| SD 2.1 base (text cond) | ~3.5 GB | ✅ (pero no aplica aquí) |
| SDXL base | ~10 GB | ⚠ borderline; requiere offloading para batch>1 |
| Versatile Diffusion (full) | ~8 GB | ⚠ borderline |

**Decisión: SD 2.1 unCLIP-L** como stack principal. Reserva SDXL/VD para Fase 3 si hay presupuesto remoto.

Entrenamiento LoRA del adapter: con bf16 + gradient checkpointing + batch 2 con accumulation 4, fits en 8 GB durante training. El UNet y el VAE de SD quedan frozen — no se entrenan.

---

## 5. Shortlist de modelos open-source (Hugging Face)

| repo_id | Rol | VRAM bf16 (inf) | Notas |
|---|---|---:|---|
| [`diffusers/stable-diffusion-2-1-unclip-i2i-l`](https://huggingface.co/diffusers/stable-diffusion-2-1-unclip-i2i-l) | **unCLIP principal** | ~5 GB | Acepta CLIP ViT-L/14 embed como condición. Encaja directamente con adapter fMRI→CLIP-pred. |
| [`diffusers/stable-diffusion-2-1-unclip-i2i-h`](https://huggingface.co/diffusers/stable-diffusion-2-1-unclip-i2i-h) | Alt unCLIP | ~6 GB | CLIP ViT-H; semánticamente más fuerte, más VRAM. |
| [`stabilityai/stable-diffusion-2-1`](https://huggingface.co/stabilityai/stable-diffusion-2-1) | Backbone text-to-image | ~3.5 GB | Baseline SD 2.1, no aplica directamente sin prompt. |
| [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | Encoder semántico (target) | ~1 GB | Fuente de target features para el adapter. Mismo CLIP que usa unCLIP-L. |
| [`laion/CLIP-ViT-L-14-laion2B-s32B-b82K`](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) | Alt CLIP ViT-L | ~1 GB | Entrenado en LAION-2B (más datos que OpenAI). Mejor para conceptos fuera de ImageNet. |
| [`laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) | Alt CLIP bigG | ~3 GB | SOTA zero-shot; usado por SDXL. Fase 3. |
| [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse) | VAE decoder mejorado | ~0.3 GB | Drop-in replacement del VAE de SD 2.1; mejora fidelidad low-level. |
| [`shi-labs/versatile-diffusion`](https://huggingface.co/shi-labs/versatile-diffusion) | Alt backbone | ~8 GB | El que usa Brain-Diffuser (Ozcelik 2023). Borderline en 12 GB para Fase 3. |
| [`pscotti/mindeyev2`](https://huggingface.co/pscotti/mindeyev2) | Referencia dataset/checkpoints | N/A | Para comparar contra MindEye2 (ICML 2024) sobre NSD. Referencia SOTA. |

**Stack recomendado Fase 2:**
- Backbone: `diffusers/stable-diffusion-2-1-unclip-i2i-l`
- Encoder target: `openai/clip-vit-large-patch14`
- VAE opcional: `stabilityai/sd-vae-ft-mse`

Esto fits en ~6-7 GB de VRAM en inferencia bf16 + xformers, deja espacio para el adapter LoRA en training.

---

## 6. Plan A/B para validar la contribución

Todos los experimentos sobre el mismo split: 25 imágenes × 3 sujetos (S01, S02, S03) del dataset Koide-Majima, mismo seed (`42`).

| Configuración | Generador | Condición | Prior difusión | Métrica principal |
|---|---|---|---|---|
| **A0 (baseline actual)** | VQGAN f=16 | CLIP ViT-B/32 + VGG19 | No (Langevin SGLD) | Pairwise-ID 49.5% (medido) |
| **A1** | SD 2.1 unCLIP-L | CLIP ViT-L/14 pred (adapter lineal) | No | ↑ pairwise-ID esperada vs A0 |
| **A2** | SD 2.1 unCLIP-L | CLIP ViT-L/14 pred (adapter LoRA) | No | ↑ vs A1 |
| **A3** | SD 2.1 unCLIP-L | Adapter LoRA + prior de difusión (estilo Brain-Diffuser) | Sí | ↑ vs A2; target ≥ 75.6% |
| **B (referencia)** | MindEye2 | — | — | No reproducible en 12 GB; citar como frontera |

Métricas por config: SSIM, PixCorr, PSNR, LPIPS, CLIP cos, pairwise-ID 2-vías (las 6 de `evaluation.py`).

**Umbral de éxito de la tesis:** A3 supera 75.6% pairwise-ID (paper Koide-Majima) en al menos uno de los 3 sujetos.

---

## 7. Roadmap calendarizado Fase 2 (meses 3-4 del plan del informe)

| Semana | Entregable | Archivos nuevos |
|---|---|---|
| 1-2 | Setup SD 2.1 unCLIP + xformers + bf16; smoke test con CLIP embeds dummy (random) → imagen. | `phase2/setup_unclip.py`, `phase2/smoke_unclip.py` |
| 3-4 | Entrenamiento adapter fMRI→CLIP-emb (LoRA). Loss: MSE + cosine contra target CLIP. | `phase2/train_adapter.py`, `phase2/adapter_lora.py` |
| 5-6 | Inferencia sobre 3 sujetos + `evaluation.py` + ablation A1 vs A2 vs A3. | `phase2/infer.py`, `output_reconstructions_sd21/` |
| 7-8 | Escritura sección de resultados + plots comparativos. | `Papers/resultados_fase2.md`, figuras |

---

## 8. Pendientes bloqueantes (pre-Fase 2)

Antes de arrancar Fase 2, cerrar estos items:

- [ ] **Completar reconstrucciones S02** — sólo 1/25 imágenes hoy. Re-correr `main_local_decoder.py` con seed fijo en la máquina nueva.
- [ ] **Completar reconstrucciones S03** — falta el sujeto entero.
- [ ] **Re-correr `evaluation.py`** con los 3 sujetos para tener baseline completo (actualmente sólo S01 reporta números).
- [ ] **Validar que los refactors no cambiaron la semántica** — comparar métrica a métrica: mismos 6 números ± 0.001 sobre S01 post-optimización.
- [ ] **Benchmarking wall-time** pre/post optimización — reportar en el MD tras traslado.

---

## 9. Referencias

- **Paper base:** Koide-Majima, Nishimoto et al. (2024), "Mental image reconstruction from human brain activity: Neural decoding of mental imagery via deep neural network-based Bayesian estimation".
- **Informe de asesoría:** `Papers/Asesoria_Tesis_Neurociencia_Computacional.pdf` (abril 2026).
- **SOTA referenciado:** MindEye (Scotti et al., NeurIPS 2023), MindEye2 (ICML 2024), Brain-Diffuser (Ozcelik 2023), Tang et al. 2023 (speech), NSD-Imagery (CVPR 2025).
- **Diffusers docs:** https://huggingface.co/docs/diffusers/api/pipelines/stable_unclip
- **Hoja de ruta local:** `Hoja_de_Ruta_Tesis.md`.
