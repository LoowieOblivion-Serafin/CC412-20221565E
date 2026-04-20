# Depuracion_Proyecto.md — Plan de limpieza + migración a NSD + Stable Diffusion

> **Contexto:** los outputs actuales (VQGAN = manchones orgánicos; SD unCLIP = carteles/texto alucinado) confirman el diagnóstico de Gemini 3.1 Pro: el dataset pre-procesado de Koide-Majima (CLIP ViT-B/32 de 512-d) está amarrado genéticamente al stack VQGAN+VGG19 de 2021. Padding 512→768 con ceros = ruido de alta frecuencia interpretado como marca de agua por SD unCLIP.
>
> **Decisión:** migrar a **Natural Scenes Dataset (NSD)** — vóxeles crudos, 8 sujetos, 73k imágenes COCO. Permite entrenar un regresor lineal fMRI→CLIP ViT-L/14 (768-d) sin el "hack" de padding, respetando la geometría de SD 2.1 unCLIP / SDXL unCLIP.
>
> **Objetivo de este documento:** tres inventarios accionables — (1) archivos a eliminar, (2) tecnologías nuevas a sumar, (3) archivos que sobreviven para la nueva arquitectura.

---

## Tabla de decisión rápida

| Categoría | Acción | Ejemplo |
|---|---|---|
| **VQGAN-specific** | ❌ Eliminar | `taming-transformers-master/`, `patch_taming.py`, `pytorch_lightning_compat.py` |
| **Features Koide-Majima 512-d** | ❌ Archivar + eliminar | `features/`, `features.tar.gz`, `mental_img_recon-main/` (excepto `ref_images/`) |
| **Documentación histórica** | 📦 Archivar (no eliminar) | `output_reconstructions/`, `Outputs Early Versions/` |
| **Stack reusable** | ✅ Conservar | `sd_decoder.py`, `evaluation.py`, `config.py` (refactor), `Papers/`, `PlantillaTesis_v01/` |
| **Infraestructura dev** | ✅ Conservar | `.gitignore`, `install.ps1`, `models_hf/` (cache HF), `requirements_py312.txt` |

---

## 1. Archivos a eliminar — justificación por archivo

### 1.1 Generador obsoleto (VQGAN + compat shims)

| Archivo | Tamaño | Razón | Acción |
|---|---:|---|---|
| `taming-transformers-master/` | 1.5 GB | Repo VQGAN 2021. Reemplazado por `diffusers.StableUnCLIPImg2ImgPipeline`. No se reutilizará ningún submódulo. | `rm -rf taming-transformers-master/` |
| `taming/` | ? | Duplicado / vendorizado parcial del anterior. | `rm -rf taming/` |
| `patch_taming.py` | 2 KB | Monkey-patch para `torch._six` (roto en PyTorch 1.9+). Único propósito: cargar VQGAN. | `rm patch_taming.py` |
| `pytorch_lightning_compat.py` | 1 KB | Shim PL 1.x ↔ 2.x requerido por `taming.models.vqgan.VQModel`. Sin VQGAN, no se usa. | `rm pytorch_lightning_compat.py` |
| `main_local_decoder.py` | 56 KB | Pipeline SGLD VQGAN-only (Fase 1). Las 6 optimizaciones aplicadas (bf16, caches, TF32) eran VQGAN-específicas. La Fase 2 usa `phase2_run_sd.py` reescrito contra NSD. | `git rm main_local_decoder.py` tras migración confirmada; mantener en historial git |

### 1.2 CLIP vendorizado (OpenAI repo original)

| Archivo | Tamaño | Razón | Acción |
|---|---:|---|---|
| `CLIP-main/` | 15 MB | Repo oficial OpenAI CLIP (vendored). Redundante: `transformers.CLIPModel` + `transformers.CLIPProcessor` ofrecen la misma API con weights en HF Hub, auto-cached, sin submódulo git. | `rm -rf CLIP-main/` después de swap a `transformers.CLIPModel` |

### 1.3 Dataset Koide-Majima 512-d (amarrado a VGG+VQGAN)

| Archivo | Tamaño | Razón | Acción |
|---|---:|---|---|
| `features.tar.gz` | 1.7 GB | Archivo comprimido del dataset pre-procesado. | ❌ Eliminar del working copy. **NO** committear — debe estar ya en `.gitignore`. Respaldar en almacenamiento externo si se necesita reproducir Fase 1. |
| `features/decoded_features/SXX/CLIP_ViT-B_32/` | ~1 GB | Features ViT-B/32 predichas fMRI→CLIP. 512-d, espacio rotado vs ViT-L/14 de SD. El "hack" de zero-pad producirá los mismos carteles. | ❌ Eliminar. Sin reemplazo directo — NSD entregará vóxeles crudos, el adapter lineal producirá los 768-d nuevos. |
| `features/decoded_features/SXX/VGG19/` | ~2 GB | Features VGG19 multi-capa. SD unCLIP **no usa VGG**. | ❌ Eliminar. |
| `features/meanDNNfeature/` | ~40 MB | Means para centrar CLIP/VGG. Específico de Koide-Majima. | ❌ Eliminar junto con `features/`. |
| `mental_img_recon-main/` (todo excepto `ref_images/`) | 11 MB | Código del paper Koide-Majima. Re-implementado en `main_local_decoder.py`. | Eliminar todo salvo `ref_images/target_images.pkl` (ground-truth de stimuli, útil para evaluación histórica S01-S03 si queremos comparar baselines). |

### 1.4 Outputs exploratorios

| Archivo | Tamaño | Razón | Acción |
|---|---:|---|---|
| `output_reconstructions/` | 14 MB | PNGs VQGAN + `metrics_*.csv` + `report_metrics.html`. Son el **baseline cuantitativo** de la tesis (pairwise-ID 49.5%). | 📦 **Archivar**, NO borrar — mover a `archive/phase1_vqgan/`. Se referenciará en la tesis como baseline contra el que mide la mejora de Fase 2. |
| `output_sd_reconstructions/` | <1 MB | Carteles alucinados del shim 512→768. No reutilizables. | ❌ Eliminar sin archivar. |
| `Outputs Early Versions/` | ? | Versiones previas exploratorias. | 📦 Archivar en `archive/early_versions/` o eliminar si ya no se referencia. |
| `phase2_run_sd.py` | 13 KB | Script con shim 512→768 que produjo los carteles. El adapter NSD reemplaza este flujo. | 🔄 **Reescribir** (no eliminar): cambiar `adapt_embedding_to_unclip` por el adapter lineal NSD→CLIP-ViT-L/14. Conservar la infra de seed, scheduler, I/O. |
| `check_setup.py` + `check_setup.txt` | 20 KB | Script de validación Fase 1 (valida VQGAN ckpt, CLIP local, etc.). | 🔄 Reescribir para validar stack nuevo: HF Hub auth, `diffusers` ≥ 0.27, NSD path, CUDA bf16. |

### 1.5 Misc

| Archivo | Razón | Acción |
|---|---|---|
| `utils_visualization.py` | Auditar: si sólo visualiza outputs VQGAN, eliminar. Si es genérico (tile, grid, matplotlib), conservar. | Leer + decidir |
| `run.ps1` (14 KB) | Orquestador Windows para Fase 1. Referencias a `main_local_decoder.py`. | 🔄 Reescribir cuando el stack NSD esté listo. |
| `._features` | Metadato macOS (fork resource). No debería estar en git. | ❌ `rm ._features` + añadir `._*` a `.gitignore`. |
| `__pycache__/` | Caché Python. | ❌ Borrar; asegurar `__pycache__/` en `.gitignore`. |

---

## 2. Tecnologías a sumar — lectura + traducción fMRI para SD+CLIP

### 2.1 Dataset nuevo

| Recurso | Propósito | URL / repo |
|---|---|---|
| **Natural Scenes Dataset (NSD)** | Vóxeles crudos 7T de 8 sujetos viendo 73k imágenes COCO. Santo grial actual. | https://naturalscenesdataset.org/ (requiere solicitud académica) |
| **NSD subset en HuggingFace** | Subset pre-procesado más liviano, listo para entrenar adapter lineal. | `pscotti/mindeyev2` (MindEye2 usa NSD) |
| **Kamitani GOD dataset** (backup) | Dataset genérico de imaginería mental (ImageNet) si NSD tarda en aprobación. | https://github.com/KamitaniLab/GenericObjectDecoding |

### 2.2 Stack de lectura/procesamiento fMRI

| Librería | Rol | Versión sugerida |
|---|---|---|
| `nibabel` | Lectura de NIfTI/GIFTI (NSD viene en estos formatos). | `>=5.0` |
| `nilearn` | Masking, ROI extraction, GLM, z-score por run. | `>=0.10` |
| `h5py` | NSD distribuye betas en HDF5. | `>=3.0` |
| `pycortex` | Visualización de vóxeles sobre superficie cortical (figuras tesis). | `>=1.2` |
| `scikit-learn` | Ridge regression + PCA para el adapter lineal (fMRI → CLIP-emb). | `>=1.3` |

### 2.3 Encoder semántico — target para el adapter

| Componente | Repo HF | VRAM |
|---|---|---:|
| CLIP ViT-L/14 (OpenAI) | `openai/clip-vit-large-patch14` | ~1 GB |
| CLIP ViT-L/14 LAION | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | ~1 GB |
| CLIP ViT-bigG/14 (para SDXL unCLIP más adelante) | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` | ~3 GB |

### 2.4 Generadores candidatos (ya en MIGRATION.md §5)

| repo_id | Rol | VRAM bf16 |
|---|---|---:|
| `diffusers/stable-diffusion-2-1-unclip-i2i-l` | **Principal** — condicionado por ViT-L/14 emb | ~5 GB |
| `stabilityai/sd-vae-ft-mse` | Mejor decoder VAE (drop-in) | ~0.3 GB |
| `shi-labs/versatile-diffusion` | Alternativa (Brain-Diffuser style) | ~8 GB |
| `pscotti/mindeyev2` | Referencia / checkpoints MindEye2 | — |

### 2.5 Adapter (Módulo 1 nuevo)

No se descarga — **se entrena localmente**. Opciones:
- **Baseline:** ridge regression sklearn (`alpha=5e4`) fMRI → CLIP-ViT-L/14. Similar a Takagi-Nishimoto 2023.
- **Upgrade:** adapter MLP + LoRA sobre embed (residual). Permite no-linealidad ligera sin overfitting.
- **SOTA-lite:** diffusion prior (DDPM pequeño sobre el residuo), estilo MindEye.

### 2.6 Logging / experiment tracking

| Librería | Rol |
|---|---|
| `wandb` o `aim` | Trackear métricas por sujeto × config del adapter. Esperable 20+ experimentos. |
| `hydra-core` | Configuración jerárquica — reemplaza el `OPTIMIZATION_CONFIG` dict de `config.py`. |

---

## 3. Lo que SOBREVIVE — stack para la migración SD + CLIP + NSD

### 3.1 Código reutilizable

| Archivo | Rol post-migración | Cambios requeridos |
|---|---|---|
| `sd_decoder.py` | Mantener tal cual. Ya entrega `load_sd_unclip_pipeline` + `reconstruct_from_embedding` con negative-prompt y CFG. | Ninguno inmediato. Cuando el adapter esté entrenado, alimentar 768-d reales (no el shim de zero-pad) — `adapt_embedding_to_unclip` será pass-through. |
| `evaluation.py` | Motor de métricas (SSIM, PixCorr, PSNR, LPIPS, CLIP cos, pairwise-ID). Agnóstico del generador. | Cambiar `original_images_path` de `target_images.pkl` (Koide-Majima, 25 stimuli) a loader NSD (COCO images referenciadas por NSD trial_id). |
| `config.py` | Router central. Mantener `PROJECT_ROOT`, `get_optimization_params`, `filter_valid_files`. | Reescribir `DATA_DIRS` → apuntar a `nsd/`. Eliminar `MODEL_PATHS` (VQGAN-specific), `MODEL_CONFIG.vgg_*`, `PROCESSING_CONFIG.dataset_structure.vgg_*`, `get_vgg_path`, `get_mean_vgg_path`. Añadir `NSD_CONFIG` con paths a betas + stimuli. |
| `requirements_py312.txt` | Base de deps. | Añadir §2.2 (nibabel, nilearn, h5py, pycortex, scikit-learn). Eliminar `taming-transformers-rom1504`, `omegaconf` (sólo para VQGAN). |

### 3.2 Infraestructura y documentación

| Archivo | Razón para conservar |
|---|---|
| `Papers/` | Biblioteca de referencia (Koide-Majima, MindEye2, NSD-Imagery, Brain-Diffuser, informe asesoría). **Crítico** para escritura tesis. |
| `PlantillaTesis_v01/` | Template LaTeX UNI con chapters, appendix, figures, bib. Destino final del documento. |
| `Hoja_de_Ruta_Tesis.md` | Roadmap formal. **Actualizar** para reflejar el cambio de dataset: Fase 2 reinicia con NSD. |
| `MIGRATION.md` | Arquitectura target (SD 2.1 unCLIP + CLIP ViT-L/14). Se mantiene, añadir sección "Pivote a NSD — abril 2026" con el racional de Gemini. |
| `README.md` + `SETUP.md` | Entry points. Actualizar instrucciones para descargar NSD + instalar stack fMRI. |
| `.gitignore` | Conservar y endurecer: añadir `features/`, `*.tar.gz`, `models_hf/`, `output_*reconstructions/`, `__pycache__/`, `._*`. |
| `install.ps1` | Instalador Windows. Actualizar deps. |
| `models_hf/` | Cache local de pesos diffusers/transformers. Ya es portable. Conservar. |

### 3.3 Archivos de referencia histórica (archivar, no eliminar)

| Ruta destino sugerida | Contenido |
|---|---|
| `archive/phase1_vqgan/metrics/` | `output_reconstructions/metrics_S01.csv`, `metrics_summary.csv`, `report_metrics.html`. Son el **baseline cuantitativo** citable en la tesis. |
| `archive/phase1_vqgan/samples/` | Subset de PNGs VQGAN (≤10) como figuras de "baseline fallido". |
| `archive/phase2_shim/` | 3 PNGs `output_sd_reconstructions/S01/*_sd_unclip.png` — evidencia visual del fallo del shim 512→768. Útil como figura en la sección "Lecciones del pivote". |
| `archive/ref_images/` | `mental_img_recon-main/ref_images/target_images.pkl` (ground-truth Koide-Majima) — si en algún momento se quiere comparar NSD vs GOD sobre los mismos stimuli. |

---

## 4. Orden de ejecución sugerido (no ejecutar aún)

```bash
# PASO 0 — snapshot antes de cualquier borrado
git checkout -b archive/phase1-vqgan-baseline
git add -A && git commit -m "snapshot: estado pre-pivote NSD (Fase 1 + shim SD)"
git push origin archive/phase1-vqgan-baseline

# PASO 1 — crear archivo histórico
mkdir -p archive/phase1_vqgan/{metrics,samples}
mv output_reconstructions/metrics_*.csv archive/phase1_vqgan/metrics/
mv output_reconstructions/report_metrics.html archive/phase1_vqgan/metrics/
# seleccionar ≤10 PNGs representativos → archive/phase1_vqgan/samples/
mv "Outputs Early Versions" archive/early_versions/ 2>/dev/null || true
mkdir -p archive/phase2_shim
mv output_sd_reconstructions/S01/*.png archive/phase2_shim/ 2>/dev/null || true
mkdir -p archive/ref_images
cp mental_img_recon-main/ref_images/target_images.pkl archive/ref_images/

# PASO 2 — eliminar VQGAN stack
rm -rf taming-transformers-master/ taming/
rm patch_taming.py pytorch_lightning_compat.py
rm -rf CLIP-main/
rm main_local_decoder.py

# PASO 3 — eliminar dataset Koide-Majima
rm -rf features/ features.tar.gz
rm -rf mental_img_recon-main/
rm -rf output_reconstructions/ output_sd_reconstructions/

# PASO 4 — limpiar metadatos
rm ._features
rm -rf __pycache__/

# PASO 5 — actualizar .gitignore
cat >> .gitignore <<'EOF'
features/
*.tar.gz
models_hf/
output_*reconstructions/
archive/
._*
__pycache__/
EOF

# PASO 6 — nueva rama de trabajo
git checkout -b pivot-nsd
git add -A && git commit -m "chore: purge VQGAN stack + Koide-Majima 512-d features"
```

**Criterio de no-retorno:** no ejecutar PASO 2+ hasta que `archive/phase1-vqgan-baseline` esté pushed al remoto.

---

## 5. Contribución de tesis redefinida (post-pivote)

La contribución original ("mejorar el pipeline Koide-Majima con SD 2.1") se refina:

> **Reconstrucción de imaginería visual desde fMRI NSD mediante Stable Diffusion 2.1 unCLIP con adapter lineal fMRI→CLIP-ViT-L/14. Comparación con baselines del SOTA 2023-2025 (Takagi-Nishimoto, MindEye/MindEye2).**

Justificación del pivote en la defensa: el experimento Fase 1 + shim produjo **evidencia negativa** de que el dataset pre-procesado Koide-Majima (512-d) es incompatible con generadores condicionados por ViT-L/14. Esto valida la decisión de migrar a un dataset que entrega vóxeles crudos.

---

## 6. Pendientes antes de empezar Fase 2 real

- [ ] Solicitar acceso NSD (https://naturalscenesdataset.org/ — requiere aplicación académica UNI).
- [ ] Ejecutar PASO 0-6 de §4 **sólo** tras confirmar acceso NSD + push remoto.
- [ ] Actualizar `Hoja_de_Ruta_Tesis.md` con nueva Fase 2 (adapter ridge + SD 2.1 unCLIP sobre NSD).
- [ ] Actualizar `MIGRATION.md` con sección "Pivote abril 2026" (racional + evidencia de los carteles).
- [ ] Escribir `Prompt_Analisis_Gemini.md` (archivo paralelo a este) para iterar el análisis con Gemini 3.1.
