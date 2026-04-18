# Hoja de Ruta y Guía para Tesis: Reconstrucción de Imaginería Visual desde fMRI

Tras un análisis riguroso de los documentos proporcionados, en particular de tu informe base sobre asesoría académica y los *papers* de referencia del estado del arte (como el modelo de *MindEye*, los avances de *Koide-Majima*, y demás modelos generativos recientes), se ha diseñado esta hoja de ruta estructurada.

> [!WARNING]
> **Brecha Crítica Identificada:** El pipeline subyacente de la tesis en su versión original (basado en VGG19 + VQGAN) ha quedado metodológicamente obsoleto frente a los avances de 2023-2025. Para que la tesis posea impacto y rigor académico, es urgente transicionar a modelos de **Difusión Latente** (Stable Diffusion) enriquecidos con priors semánticos como **CLIP**.

---

## 1. Definición Estratégica (Scope)

Para garantizar la viabilidad y alta calidad del proyecto con los recursos de software y hardware (RTX 3070, 8GB VRAM) actuales, hay que focalizar todo el esfuerzo eliminando la dispersión temática (como la decodificación del habla o problemas abstractos de complejidad NP).

**Título de Tesis Propuesto:**
> *Reconstrucción de Imaginería Visual desde Señales fMRI mediante Modelos de Difusión Latente con Prior Semántico CLIP: Una Mejora al Marco Bayesiano de Koide-Majima*

---

## 2. Cambios Metodológicos para Mejorar Resultados

La adopción de mejores resultados vendrá dictaminada directamente por la re-estructuración de la metodología hacia los estándares avalados por los papers revisados (*Brain-Diffuser*, *Koide-Majima (2024)*, *MindEye*):

1. **Sustitución del Generador Neural:** Reemplazar el VQGAN por modelos generativos del estado del arte como **Stable Diffusion 2.1** o un **Versatile Diffusion** ligero, operando con _Low-Rank Adaptation_ (LoRA) para evitar el colapso de la memoria VRAM de la GPU.
2. **Prior Semántico (CLIP):** El decodificador no debe converger al espacio de píxeles, sino predecir las representaciones semánticas en el espacio CLIP (ej. ViT-L/14). Esto resuelve el llamado *modality gap*.
3. **Optimización Biológica de Vóxeles:** Utilizar una partición jerárquica de la corteza visual (V1/V2 para bordes, LOC para semántica) y modelos como regresión L2 (Ridge) antes de que la red procese las señales completas para sortear dimensionalidades masivas.

---

## 3. Hoja de Ruta de Implementación (Cronograma de 6 Meses)

Esta es la ruta de investigación empírica a ejecutar paso por paso para obtener resultados publicables:

### Fase 1: Puesta a Punto y Baseline (Semanas 1-4)
- **Datos:** Descargar y preprocesar el volumen de datos de Koide-Majima provisto en Figshare o el subset preprocesado de NSD (Natural Scenes Dataset).
- **Librerías:** Configurar un entorno en PyTorch nativo (versión 2.x), usando `nilearn` y `nibabel` para tratar los archivos `.nii`.
- **Reproducibilidad:** Reproducir de manera honesta el sistema previo de *Koide-Majima et al. (2024)* como tu métrica base sobre la cual probarás tu mejora (Aspirar al ~75.6% de identificación *pairwise* actual).

### Fase 2: Implementación de la Nueva Arquitectura (Meses 2-3)
- **Modelamiento:** Entrenar un decodificador base de características cerebrales (MLP ligero o Regresión Ridge) para alinear las respuestas BOLD (fMRI) al espacio latente CLIP.
- **Generación:** Condicionar el modelo pre-entrenado de *HuggingFace Diffusers* a dichos embeddings para que el sistema de difusión construya imágenes coherentes.
- **Micro-Optimización:** Aplicar rutinas de `gradient checkpointing` y formato bfloat16 a lo largo del proceso para que toda la iteración quepa holgadamente dentro de tu RTX 3070 local.

### Fase 3: Evaluación y Resultados (Meses 4-5)
Para que el trabajo tenga crédito válido, debes medir y comparar rigurosamente:
- **Correlación de Píxel y Similitud Estructural (SSIM)** para la precisión espacial (under-level).
- **Distancia de CLIP y Precisión _Pairwise_ ID** para medir qué tan acertada es semánticamente la imagen generada comparada con la imagen de estímulo real original.
- **Ablación:** Crea un apartado de contraste experimentando una métrica "VQGAN vs. Modelo de Difusión". Una mejora metodológica comprobada de incluso +3% bajo métricas objetivas será más que suficiente para garantizar el triunfo del proyecto.

---

## 4. Guía Maestra de Redacción para la Tesis de Grado

La tesis no debe leerse como un compendio de códigos, sino como un reporte lógico, autocrítico y progresivo. Sigue esta pauta de capítulos recomendada:

### Capítulo 1: Motivación y Avances del Estado del Arte
- Sitúa tu trabajo inmediatamente en el umbral post-2023. Enumera claramente los éxitos tempranos con GANs, para después destruir sus desventajas citando a *Tang et al.* y *Scotti et al. (MindEye)*, dejando el terreno servido para tu justificación en favor de la difusión latente.

### Capítulo 2: Marco Teórico y Limitaciones Algorítmicas
- Resume brevemente cómo la información BOLD fluye por la corteza y por qué inferir este conjunto es parecido a un problema de complejidad computacional dura (Menciona superficialmente *3-Partition* solo como un justificativo metodológico a por qué aplicas reducción de dimensionalidad PCA en fMRI). 

### Capítulo 3: Propuesta Metodológica y Diseño 
- Elabora un diagrama macro del _Pipeline_. 
- Se sumamente iterativo explicando los recortes e ingenierías (uso de LoRA/bfloat16) que tuviste que aplicar para viabilizar el proyecto bajo las fronteras del hardware albergado por la computadora de la tesis.

> [!TIP]
> **Ser Inquebrantable en la Honestidad Tecnológica**
> Otorga una altísima prioridad a registrar detalladamente qué procesaste localmente y qué obtuviste del *transfer learning*. Todos los tribunales neurocientíficos valoran la transparencia empírica infinitamente más que resultados sobreajustados e inauditos.

### Capítulo 4: Análisis de Resultados y Experimentación
- Separa la validación en **"Imágenes Percibidas"** (cuando el usuario miró algo en la prueba médica) vs **"Imágenes Imaginadas/Visual Imagery"**.  
- Enseña aquí tu *Ablation Study* mediante tablas numéricas comparativas y gráficas de tus métricas (IS, Pairwise ID, SSIM).

### Capítulo 5: Limitaciones Acotadas y Trabajos Futuros (Conclusión)
- Refuerza por qué este logro local es el "Paso 1" obligatorio rumbo a la utopía teórica a largo plazo del descifrado sináptico visual continuo en _streaming/tiempo real_ de BCI.
