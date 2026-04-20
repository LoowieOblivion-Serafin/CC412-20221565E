# PASO 0 - Snapshot
git add -A
git commit -m "snapshot: estado pre-pivote NSD (Fase 1 + shim SD y logs de Gemini)"
git checkout -b archive/phase1-vqgan-baseline
git push origin archive/phase1-vqgan-baseline

# PASO 1 - Archivo historico
New-Item -ItemType Directory -Force -Path "archive/phase1_vqgan/metrics" > $null
New-Item -ItemType Directory -Force -Path "archive/phase1_vqgan/samples" > $null
Move-Item -Path "output_reconstructions/metrics_*.csv" -Destination "archive/phase1_vqgan/metrics/" -Force -ErrorAction SilentlyContinue
Move-Item -Path "output_reconstructions/report_metrics.html" -Destination "archive/phase1_vqgan/metrics/" -Force -ErrorAction SilentlyContinue

Move-Item -Path "Outputs Early Versions" -Destination "archive/early_versions" -Force -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path "archive/phase2_shim" > $null
Move-Item -Path "output_sd_reconstructions/S01/*.png" -Destination "archive/phase2_shim/" -Force -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path "archive/ref_images" > $null
Copy-Item -Path "mental_img_recon-main/ref_images/target_images.pkl" -Destination "archive/ref_images/" -Force -ErrorAction SilentlyContinue

# PASO 2 - Eliminar VQGAN stack
Remove-Item -Recurse -Force "taming-transformers-master", "taming", "CLIP-main" -ErrorAction SilentlyContinue
Remove-Item -Force "patch_taming.py", "pytorch_lightning_compat.py", "main_local_decoder.py" -ErrorAction SilentlyContinue

# PASO 3 - Eliminar dataset
Remove-Item -Recurse -Force "features", "mental_img_recon-main", "output_reconstructions", "output_sd_reconstructions" -ErrorAction SilentlyContinue
Remove-Item -Force "features.tar.gz" -ErrorAction SilentlyContinue

# PASO 4 - Limpiar metadatos
Remove-Item -Force "._features" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "__pycache__" -ErrorAction SilentlyContinue

# PASO 5 - Actualizar .gitignore
$gitignoreContent = @"

# Pivot NSD Ignore Rules
features/
*.tar.gz
models_hf/
output_*reconstructions/
archive/
._*
__pycache__/
"@
Add-Content -Path .gitignore -Value $gitignoreContent

# PASO 6 - Nueva rama de trabajo
git checkout -b pivot-nsd
git add -A
git commit -m "chore: purge VQGAN stack + Koide-Majima 512-d features"
