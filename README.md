# 🧠 Reconstrucción de Imágenes Mentales desde Actividad Cerebral (fMRI a Stable Diffusion)

Proyecto que convierte señales fMRI del cerebro humano en imágenes visuales reconstruidas utilizando Modelos de Difusión.

## Descripción

Este proyecto forma parte de una investigación de tesis centrada en la decodificación neuronal. Su objetivo es "leer" características visuales directamente desde la actividad cerebral de un sujeto humano y reconstruirlas visualmente empleando modelos de generación de estado del arte.

**Evolución del Proyecto (Fase 2):**
Anteriormente, el pipeline (Fase 1) dependía de la arquitectura bayesiana de Koide-Majima et al. (2024), empleando VQGAN + VGG19. Sin embargo, su desempeño se veía severamente obstaculizado, operando casi a niveles de azar. En la **Fase 2 actual**, hemos migrado radicalmente hacia la arquitectura **Stable Diffusion 2.1 unCLIP**. Ahora, los tensores extraídos del córtex visual se introducen como embebimientos condicionantes directos en el espacio latente de difusión, incrementando drásticamente el fotorrealismo y la coherencia semántica.

## Características Principales

✅ **Arquitectura unCLIP** - Condicionamiento directo vía CLIP embeddings (espacio latente fMRI).  
✅ **Priorización Semántica** - Moduladores textuales de mitigación de padding gaussiano integrados para estabilidad de red UNet.  
✅ **Optimización RTX 4070 Ti** - Uso estricto de atenciones `xformers` y tensores `bfloat16`.  
✅ **Reproducibilidad Rigurosa** - Semillas bloqueadas globalmente para inferencia bit-exacta.  
✅ **Compatibilidad Multi-Sujeto** - Pipelines de procesamiento masivo para S01, S02 y S03.  

## Diferencias con la Baseline (VQGAN original)

- 🚀 **Desempeño:** SD 2.1 unCLIP resuelve el colapso visual ("manchones abstractos") del optimizador Langevin.
- ⚙️ **Velocidad:** Inferencia paralela, en contraposición a las extremas 1000 iteraciones requeridas por imagen en el optimizador iterativo anterior.
- 🔬 **Validación Científica:** Uso de CFG controlada e inyección calculada de ruido espacial (`noise_level`).

## Requisitos del Sistema

- **Python**: 3.12 (estable, recomendado).
- **RAM**: 16 GB recomendado.
- **GPU**: Tarjeta gráfica NVIDIA (Optimizado en RTX 4070 Ti, 12 GB VRAM).
- **Espacio**: ~15GB para entorno condensado (Modelos SD 2.1 unCLIP + Datasets HF y Tensores fMRI).

## Estructura del Proyecto

```text
ACECOM-Project/
├── features/                    # Dataset fMRI Base 
│   ├── decoded_features/        # Vectores pre-decodificados de S01, S02, S03 (ViT-B/32)
├── output_sd_reconstructions/   # Carpeta generada auto con las inferencias unCLIP
├── main_local_decoder.py        # [LEGACY] Script de la baseline original VQGAN
├── sd_decoder.py                # [CORE] Implementación del Pipeline SD 2.1 unCLIP
├── phase2_run_sd.py             # [SCRIPT] Main entry-point para inicializar inferencia
├── evaluation.py                # [METRICS] Cálculo de PixCorr, SSIM, LPIPS y Pairwise
├── config.py                    # Configuración estática
└── MIGRATION.md                 # Informe técnico de la justificación arquitectónica
```

## Quick Start

### Instalación Automática
Por favor, asegúrate de estar operando en un entorno local y consulta la **[Guía Completa en SETUP.md](SETUP.md)** para una configuración de librerías seguras (`xformers`, `diffusers`).

### Inferencia: Fase 2
Para evaluar los features embebidos en el modelo de difusión, ejecuta el puente de inferencia end-to-end:

```bash
# Procesa todos los sujetos (S01, S02, S03):
python phase2_run_sd.py

# Smoke test (solo valida el primer sujeto, 3 imágenes máximo):
python phase2_run_sd.py --subjects S01 --limit 3
```

> **NOTA SOBRE DIMENSIONALIDAD**: El pipeline actualmente realiza un "shim padding" de los features extraídos (CLIP ViT-B/32 de 512d) a dimensionalidad ViT-L/14 (768d), que es requerida por SD 2.1 unCLIP.

## Licencia & Créditos

Este código es el resultado integral de una investigación de tesis de la Universidad Nacional de Ingeniería (UNI) (2025-2026) llevada a cabo por Alvaro Taipe. El diseño de la migración toma como base empírica refactorizar y modernizar los resultados exploratorios de Koide-Majima et al. (2024).

---

**[Ir al manual de SETUP.md para instrucciones de entorno técnico]**
