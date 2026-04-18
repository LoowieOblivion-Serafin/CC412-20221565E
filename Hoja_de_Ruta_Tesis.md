# Proyección Estructural de Investigación: Reconstrucción de Imágenes Mentales
*(Versión ajustada a la plantilla de Proyecto de Tesis `main.pdf`)*

> Este documento transpone sistemáticamente la estrategia metodológica de tu asesoría a los Capítulos ($Cap1 \dots Cap7$) definidos en el documento maestro de tu tesis, amalgamando la validación cuantitativa del baseline como sustento del problema.

---

## Título Oficial Propuesto
*Reconstrucción de Imaginería Visual desde Señales fMRI mediante Modelos de Difusión Latente con Prior Semántico CLIP: Una Mejora al Marco Predictivo Lineal*

---

## Estructura Capitular y Redacción

### Capítulo 1: Introducción y Planteamiento del Problema
**Misión:** Establecer empíricamente la obsolescencia metodológica de los modelos GAN para imaginería visual y presentar tu solución.
* **Planteamiento Contextual:** A partir de 2023, la comunidad computacional demostró que la reconstrucción fMRI requiere separar las directrices perceptuales de las semánticas (estado del arte).
* **Definición del Problema Cuantitativo (Resultados Base - $S_{01}$):** Argumentación del diagnóstico algorítmico utilizando tus propios hallazgos extraídos. Al replicar el *baseline* clásico *(VGG19+VQGAN)* se evidencia un rendimiento estocástico en identificación semántica (Pairwise ID de $49.5\%$, virtualmente azar). Adicionalmente, el índice PixCorr decreciente ($-0.027$) y el elevado LPIPS ($0.734$) confirman que la red genera anomalías espaciales inidentificables.
* **Justificación de la Solución:** La deficiencia de $50\%$ hallada contra el óptimo del estado del arte ($75.6\%$) sustenta incontestablemente la necesidad de migrar el pipeline a Difusión Latente condicionado con codificadores CLIP (ViT-L).

### Capítulo 2: Marco Teórico y Limitaciones
**Misión:** Sintetizar la robustez teórica subyacente y sentar base para la experimentación.
* **El Modality Gap:** Profundización de cómo las distancias semánticas en el cerebro humano ($fMRI$) no se correlacionan trivialmente con la semántica digital del modelo fundacional, requiriendo un puente conectivo.
* **Difusión Latente (LDMs):** Sustento matemático sobre cómo los modelos de ruido-supresión (*Stable Diffusion*) resuelven estructuralmente el problema perceptual sin colapsar geométricamente a diferencia de las arquitecturas previas.
* **Análisis de Complejidad Computacional (Límite Acotado):** Exposición de cómo seleccionar combinatoriamente un subconjunto óptimo de *k* vóxeles de toda la corteza se cataloga como $NP\text{-}Hard$, justificando tu elección de partición biológica jerárquica ($V1/V2$ para bordes, $LOC/FFA$ para semántica explícita).

### Capítulo 3: Propuesta Metodológica y Arquitectura de Solución
**Misión:** Describir la ingeniería de integración paso a paso y la infraestructura manejada por el software.
* **Tratamiento del Archivo de Vóxeles:** Uso de máscaras sobre archivos $NIfTI$ para recuperar los Tensores Corticales visuales (Librería `Nilearn`).
* **Decodificador Ridge / Reducción Fisiológica:** Algoritmos para entrenar pre-mapeadores ligeros que inyecten el estimulo cerebral al espacio latente compartido.
* **Optimizaciones de Hardware Local (Ingeniería de VRAM):** Enfoque estricto documentando el uso ingenieril de Low-Rank Adaptation (LoRA), gradientes segmentados (`gradient checkpointing`) y `bfloat16`. Esto prueba tu destreza para ejecutar lógicas masivas en una *RTX 4070 Ti* saltando los límites que en teoría demandarían servidores empresariales A100.

### Capítulo 4: Fases de Experimentación y Desarrollo SOTA
**Misión:** La narración de la ejecución técnica (el núcleo temporal trazado para tu investigación de Tesis 1).
* **Recolección y Armonización:** Integración del preprocesamiento formal con los Set de datos extraídos (ej. *Natural Scenes Dataset - NSD* o los datos base de *Koide-Majima*).
* **Experimentos de Ablación y Entrenamiento:** Descripción de la rama en GitHub donde se gestó la generación (*rtx4070ti-execution*) donde el decodificador reemplaza iterativamente predicciones nulas por predicciones semánticamente guiadas por CLIP.

### Capítulo 5: Resultados, Métricas Numéricas y Discusión
**Misión:** Tabulación de la mejora. Tu meta final evaluativa en contraposición con tu baseline.
* **Evaluación Bidimensional:** Reportes evaluados sistemáticamente mediante algoritmos implementados estandarizados de *evaluation.py*. 
* **Tablas de Criterio Semántico vs Estructural:**
  1. *Estructura Perceptual:* SSIM y PixCorr evidenciando la recuperación formal de las formas lógicas en las iteraciones de la difusión.
  2. *Retención Conceptual:* Inception Score (IS) y el crucial "2-way Pairwise-ID" probando qué tan lejos subieron las estadísticas del azar del $49.5\%$.
* **Discusión de Anomalías:** Confrontar las predicciones sobre la imaginería visuoespacial (*Imagery*) versus la vista pasiva.

### Capítulo 6 (y opcional 7): Conclusiones y Trabajo Futuro Integral
* **Síntesis del Precedente:** Confirmar que los modelos generativos contemporáneos resuelven la estocasticidad descodificatoria.
* **Proyección (Long-Term Goal):** Definir tus resultados en esta investigación (*Reconstrucción offline*) como el primer peldaño validado en el camino utópico hacia interfaces Cerebro-Máquina (BCI) proyectadas para un descifrado de pensamiento cinético o visual en tiempo real.
