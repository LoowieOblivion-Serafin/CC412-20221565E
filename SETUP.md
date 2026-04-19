# 🛠️ Guía de Instalación y Configuración (Fase 2 - Tesis)

Esta guía documenta la infraestructura tecnológica requerida para inicializar y desplegar la nueva arquitectura de **Stable Diffusion 2.1 unCLIP**, que ha reemplazado oficialmente el entorno heredado de VQGAN. 

Para lograr tiempos de inferencia instantáneos y no desbordar (Out of Memory - OOM) la tarjeta gráfica (12GB VRAM en la RTX 4070 Ti), esta configuración exige estricta atención a la instalación de PyTorch y Xformers.

---

## 1. Requisitos de Hardware y Software

- **GPU Objetivo**: NVIDIA con 12 GB VRAM (RTX 3060/4070 Ti etc.) con arquitectura compatible.
- **Python**: **3.12** puro (Recomendamos evitar arquitecturas base Anaconda para evitar conflictos de pathing conda en Windows si no se controla adecuadamente).
- **Controladores**: CUDA Toolkit 12.1 o superior.
- **Memoria de Almacenamiento**: Mínimo de 15GB libres en SSD (SD 2.1 unCLIP pesa ~5GB al instanciarse en caché).

---

## 2. Preparación del Entorno (Windows)

Si partes desde cero, crea un entorno virtual (venv) para aislar las variables del entorno del sistema:

```powershell
# Instanciar el entorno
python -m venv env_tesis

# Activar el entorno
.\env_tesis\Scripts\Activate.ps1
```

---

## 3. Instalación Base: PyTorch + CUDA

SD 2.1 unCLIP usa redes convolucionales extensas. Debemos asegurar que PyTorch corra la compilación correcta local contra los Tensor Cores de NVIDIA.

Ejecuta lo siguiente para empalmar con **CUDA 12.1**:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Verificación Obligatoria de Hardware:
Ejecuta esto para asegurarte que estás atado a la gráfica de renderizado, y no al procesador, algo que destruiría el tiempo de vida de la investigación:

```powershell
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

---

## 4. Instalación de Difusión (Diffusers + Xformers)

Los núcleos estables de reconstrucción precisan librerías nativas publicadas por HuggingFace.

Para instalar todos los dependientes explícitos de tu `requirements_py312.txt`:

```powershell
pip install -r requirements_py312.txt
```

> **📌 NOTA SOBRE XFORMERS**: 
> Este entorno está diseñado para descargar `xformers` (incluido en los _requirements_). Esta librería inyecta una _"Atención Eficiente en Memoria"_ (Memory Efficient Attention) en el modelo de Stable Diffusion. 
> Gracias a esto, el sistema comprimirá el pico lógico de uso en inferencia desde 9 GB de VRAM a escasos 5 GB, facilitando cargas extra.

---

## 5. Estructura y Datasets Obtenidos

Tu ecosistema actual del root del repositorio debe parecerse a este formato:

```text
ACECOM-Project/
├── features/                    # Data inicial sin procesar (descargar de drive).
│   ├── decoded_features/
│   │   ├── S01/                 # ~26 tensores viT-B_32 por paciente
│   │   ├── S02/ 
│   │   └── S03/ 
├── models_hf/                   # (Auto-generada) Modelos HF cacheados offline
├── output_sd_reconstructions/   # (Auto-generada) Destino de Inferencia fMRI -> Imágenes
├── sd_decoder.py                # Red Difusora
└── phase2_run_sd.py             # Runtime principal
```

*Importante:* A diferencia del módulo Fase 1 heredado, **ya no requerimos clonar el repositorio de OpenAI CLIP ni el Taming Transformers**. `Diffusers` administra los modelos en una capa de caja negra estandarizada que se descarga solitaria a la carpeta `/models_hf/`.

---

## 6. Ejecución y Validaciones de Estado (Smoke Checks)

Para asegurar que los Tensores de cerebros son compatibles de leer y alimentar a la Red de Difusión y asegurar la estabilidad de la semántica de la red UNet mediante los *Priors de estabilización gaussianos*. 

Prueba este comando que limita la generación a sólo 3 imágenes para confirmar que no ocurran cuelgues térmicos o faltas de módulos.

```powershell
python phase2_run_sd.py --limit 3 --subjects S01
```

Una corrida inicial exitosa descargará los pesos (`fp16/safetensors`) del HuggingFace Hub, inicializará el programador `DPMSolverMultistepScheduler` (por defecto fijado a solo 25 pasos), e inyectará la semilla global en la matemática tensorial, escupiendo algo como:
`[S01] (1/3) imagen_x.pkl -> S01_imagen_x_sd_unclip.png`

**Ejecución Completa (Inferencia Total)**
Una vez verificado, ejecuta el dataset entero (para S01, S02, S03):

```powershell
python phase2_run_sd.py
```

### Problemas Frecuentes:

1. **El output parece "Texto / ruido periódico difuminado"**
Esto comúnmente ocurría si intentabas introducir tensores de fMRI mapeados a dimensionalidad `512` explícitamente sobre el UNet sin el "padding semántico inverso" o inyecciones de Priors como base textuales. Revisa el valor `noise_level` (ideal 250) en el módulo de reconstrucción, que actualmente ya está aplicado por defecto.

2. **La advertencia `SHIM ACTIVO` en consola**
Tu dataset original proyectó a espacios de características genéricos `ViT-B/32`, el modulo StableUnClip requiere `ViT-L/14` explícito. Esta bandera asume la traducción usando empaquetados espaciales, es intencional y vital para esta fase.
