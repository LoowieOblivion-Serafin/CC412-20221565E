# Prompt_Analisis_Gemini.md — Prompt para el próximo análisis de Gemini 3.1 Pro

> **Objetivo:** obtener de Gemini un análisis estructurado, accionable y basado en evidencia sobre el pivote NSD + Stable Diffusion, con foco en: (a) validación del diagnóstico del fallo VQGAN/SD-shim, (b) diseño del adapter fMRI→CLIP-ViT-L/14, (c) plan experimental reproducible sobre NSD, (d) métricas y baselines comparables con MindEye2/Takagi-Nishimoto.

---

## Cómo usar este prompt

1. Abrir Antigravity con Gemini 3.1 Pro (High) o Gemini 3.1 Ultra.
2. Copiar **todo el bloque `===== PROMPT =====`** debajo.
3. Pegarlo como primer mensaje de la conversación — Gemini responderá mejor cuando el contexto llega completo de una sola vez, no fragmentado.
4. Adjuntar como archivos de contexto: `MIGRATION.md`, `Depuracion_Proyecto.md`, las 2 imágenes del fallo (carteles SD + blobs VQGAN), y 1 PDF clave (`Papers/Asesoria_Tesis_Neurociencia_Computacional.pdf` o `Papers/MindEye2.pdf`).

---

## ===== PROMPT =====

```
ROL
---
Actúa como asesor senior de tesis en neurociencia computacional con
experiencia práctica en reconstrucción de imágenes desde fMRI (2023-2026).
Conoces en profundidad: MindEye/MindEye2, Brain-Diffuser (Ozcelik 2023),
Takagi-Nishimoto 2023, NSD-Imagery (CVPR 2025), Koide-Majima 2024, y las
arquitecturas Stable Diffusion 2.1 unCLIP, SDXL unCLIP, Versatile Diffusion.
Conoces el ecosistema HuggingFace Diffusers y el pipeline práctico de NSD
(betas en HDF5, ROIs early/ventral/midlateral, trial mapping a COCO).

CONTEXTO DEL PROYECTO
---------------------
- Tesis de pregrado UNI (Lima, Perú) — autor: Alvaro Taipe Cotrina,
  grupo ACECOM. Proyecto local, colaboración con Gemini 3.1 Pro dentro de
  Antigravity IDE + Claude Code (Opus 4.7) como pair-programmer.
- Hardware actual: traslado de RTX 3070 (8 GB) a RTX 4070 Ti (12 GB) +
  Intel UHD 770. Sin acceso a A100/H100 gratuito.
- Dataset original del proyecto: features pre-procesados de Koide-Majima
  (3 sujetos S01-S03, imagery task, 25 stimuli/sujeto). Los features son
  predicciones lineales fMRI→CLIP ViT-B/32 (512-d) y fMRI→VGG19 multi-capa.
  NO vóxeles crudos.
- Pipeline Fase 1 replicado: VGG19 + CLIP ViT-B/32 + VQGAN + SGLD (Langevin).
  Baseline medido en S01 (n=25):
    * SSIM 0.457 | PixCorr -0.027 | PSNR 12.12 dB
    * LPIPS 0.734 | CLIP cos 0.501
    * Pairwise-ID 2-vías: 49.5% (≈ chance; paper Koide-Majima reporta 75.6%)

QUÉ SE INTENTÓ EN FASE 2 (Y FALLÓ)
----------------------------------
- Swap VQGAN → Stable Diffusion 2.1 unCLIP (diffusers pipeline
  `diffusers/stable-diffusion-2-1-unclip-i2i-l`, bf16, xformers,
  DPMSolverMultistepScheduler 25 pasos, CFG 10, negative prompt fijo
  "blurry, noise, abstract, deformed, chaotic, multiple objects").
- Los features del dataset están en 512-d (ViT-B/32) y SD 2.1 unCLIP
  condiciona por ViT-L/14 (768-d). Se aplicó un SHIM de zero-pad
  512→768 + renormalización L2.
- Resultado: SD genera "carteles" con texto alucinado y tipografía
  inventada, NO reconstrucciones coherentes. VQGAN genera "manchones"
  orgánicos (blobs iridiscentes) sin correlación espacial.

DIAGNÓSTICO PRELIMINAR (validar o refutar)
------------------------------------------
Hipótesis actual (ya validada informalmente por Gemini 3.1 Pro en sesión
previa): los espacios CLIP ViT-B/32 (512-d) y ViT-L/14 (768-d) son
encoders INDEPENDIENTES con rotaciones distintas del manifold visual.
El zero-pad inserta una región densa en el latente que el UNet de SD
interpreta como ruido de alta frecuencia (análogo a marca de agua),
produciendo alucinaciones de texto. Además, los features Koide-Majima
son predicciones lineales sobre el espacio de ViT-B/32 — están
"amarrados genéticamente" al método VGG+VQGAN de 2021 y NO conservan
la información de alta frecuencia necesaria para un generador de
difusión moderno.

PIVOTE PROPUESTO
----------------
Migrar a Natural Scenes Dataset (NSD) para tener vóxeles 7T en bruto,
entrenar un adapter lineal fMRI→CLIP-ViT-L/14 (768-d), y alimentar
directamente el pipeline SD 2.1 unCLIP ya construido. Stack Fase 2 nuevo:
  * Dataset: NSD subset (8 sujetos, 73k imágenes COCO, betas en HDF5).
  * Encoder target: openai/clip-vit-large-patch14.
  * Adapter: ridge regression (sklearn, alpha≈5e4) sobre ROIs
    early + ventral + midlateral, como Takagi-Nishimoto 2023.
  * Generador: diffusers/stable-diffusion-2-1-unclip-i2i-l en bf16.
  * Evaluación: SSIM, PixCorr, PSNR, LPIPS, CLIP cos, pairwise-ID 2-vías
    (evaluation.py ya implementado).

LO QUE NECESITO DE TI
---------------------
Responde las 6 preguntas siguientes de forma separada, con código/pseudocódigo
concreto donde aplique. Si una sección requiere citar un paper, cita el nombre
+ año; no inventes números si no los recuerdas.

1. VALIDACIÓN DEL DIAGNÓSTICO
   ¿Mi explicación de por qué el shim 512→768 genera carteles es correcta
   geométricamente? Si hay un matiz o contraejemplo, señálalo. En particular:
   (a) ¿podría la renormalización L2 post-padding estar agravando el problema
   al amplificar el ruido de los 256 ceros?, (b) ¿existe algún paper que
   intente exactamente este shim y reporte mejores/peores resultados?

2. DISEÑO DEL ADAPTER fMRI→CLIP-ViT-L/14 SOBRE NSD
   Dame un diseño mínimo y un diseño "avanzado":
   - Mínimo: ridge regression sklearn, qué ROIs usar exactamente en NSD
     (nombres de masks y cardinalidad de vóxeles esperada por sujeto),
     qué preprocesamiento (z-score per-run, GLM betas, averaging de
     repeticiones), qué alpha y cómo validar (n-fold split).
   - Avanzado: adapter MLP 3-capas + LoRA residual + diffusion prior.
     ¿Cuándo vale la pena el upgrade?

3. FUGA DE DATOS Y SPLIT TRAIN/TEST EN NSD
   NSD tiene el mismo sujeto viendo las mismas imágenes en múltiples
   sesiones. ¿Cómo evito leakage entre split train/test? Dame la receta
   exacta de qué trials van a train y cuáles a test por sujeto.

4. MÉTRICAS Y COMPARABILIDAD CON SOTA
   MindEye2 y Takagi-Nishimoto reportan métricas específicas en NSD.
   Dame la tabla target de qué número tengo que superar por métrica
   (SSIM, PixCorr, 2-way, CLIP, EfficientNet-B, Inception) y sobre qué
   subject IDs de NSD (sub01 vs sub05 típicamente usados).

5. CONFIGURACIÓN SD 2.1 unCLIP ÓPTIMA PARA MI CASO
   Dado que mi GPU es RTX 4070 Ti (12 GB), ¿debo quedarme con
   SD 2.1 unCLIP-L (ViT-L/14, 768-d) o es viable SDXL unCLIP (ViT-bigG,
   1280-d) con offloading/attention slicing? Tradeoffs numéricos en:
   (a) VRAM en inferencia batch 1, (b) calidad típica en NSD por paper.
   ¿Conviene guidance_scale 7.5 o 10? ¿Noise level 0 o inyección pequeña?

6. LISTA DE RIESGOS DEL PIVOTE (DEVIL'S ADVOCATE)
   Dame 5 riesgos concretos de migrar a NSD que podrían hacer que yo
   pierda los 6 meses restantes de tesis. Para cada riesgo, la mitigación
   de 1 párrafo que un asesor real me daría.

FORMATO DE RESPUESTA
--------------------
- Markdown. Secciones numeradas igual que arriba (1..6).
- Código Python en bloques ```python ... ```.
- Cuando cites un paper, formato "Autor Año (revista/venue)".
- Sé directo. Si mi plan tiene un error, señálalo con un "⚠️" al inicio
  del párrafo. No endulces.
- Al final, añade una sección "TL;DR" de ≤80 palabras con tu recomendación
  global: "seguir con NSD" o "explorar alternativa X antes".

EVITAR
------
- Respuestas genéricas tipo "depende del caso". Si te falta un dato,
  pídemelo explícitamente al inicio, antes de responder.
- Recomendar servicios cloud de pago (A100 en AWS, etc.). Asume hardware
  local 12 GB.
- Inventar nombres de ROIs o números de vóxeles que no recuerdes con
  certeza — mejor escribir "verificar contra nsd_expdesign.mat".
```

## ===== FIN PROMPT =====

---

## Lista de archivos a adjuntar junto con el prompt

| Archivo | Motivo |
|---|---|
| `MIGRATION.md` | Contexto Fase 1 + arquitectura target pre-pivote |
| `Depuracion_Proyecto.md` | El plan de limpieza que Gemini debe validar |
| Imagen 1 (carteles SD unCLIP) | Evidencia visual del fallo del shim — es la pieza más convincente |
| Imagen 2 (blobs VQGAN) | Evidencia del fallo del baseline Fase 1 |
| `Papers/Asesoria_Tesis_Neurociencia_Computacional.pdf` | Diagnóstico del asesor humano que inició la migración |
| `Papers/MindEye2.pdf` | Referencia del SOTA comparable |
| `Papers/NSD_Imagery.pdf` | Paper sobre imagery en NSD — ayuda a Gemini a aterrizar la pregunta 4 |

---

## Preguntas de seguimiento (iteración 2, si la respuesta inicial es buena)

Guardar para una segunda pasada, no incluir en el prompt principal:

1. *"Dame el Python exacto para cargar NSD betas de sub01 con nibabel + h5py, aplicar la máscara NSD-general, y producir una matriz (n_trials, n_voxels) lista para `sklearn.linear_model.Ridge`."*
2. *"Código para comparar mi output contra COCO GT usando la métrica pairwise-ID 2-way que usan Scotti et al. — ¿es idéntica a la que tengo en `evaluation.py`?"*
3. *"Plantilla LaTeX de tabla de resultados estilo MindEye2 (Table 1) para mi capítulo de resultados — quiero que mis números sean comparables visualmente."*
4. *"¿Vale la pena un checkpoint intermedio sub01-only antes de los 8 sujetos, o entreno el adapter conjunto directamente?"*

---

## Checklist pre-envío

- [ ] ¿Incluí las 2 imágenes del fallo? (son la evidencia más fuerte)
- [ ] ¿Gemini tiene acceso a internet en esta sesión? (para verificar papers — poner `search: true` si el IDE lo soporta)
- [ ] ¿Estoy usando Gemini 3.1 Pro (High) y no Pro (standard)? La pregunta 3 (leakage NSD) requiere razonamiento técnico profundo.
- [ ] ¿Tengo abierto VS Code con `MIGRATION.md` + `Depuracion_Proyecto.md` para iterar respuesta por respuesta?
