# =====================================================================
# run.ps1 - Script de ejecucion del Pipeline ACECOM
# Reconstruccion de Imagenes Mentales desde Actividad Cerebral
#
# USO: .\run.ps1
# =====================================================================

$ErrorActionPreference = "Continue"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  PIPELINE ACECOM - Reconstruccion de Imagenes Mentales" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# ─────────────────────────────────────────────────────────────────────
# PASO 1: Detectar el intérprete Python correcto
# ─────────────────────────────────────────────────────────────────────
Write-Host "[1/5] Detectando interprete Python..." -ForegroundColor Yellow

$PYTHON = $null

$candidates = @(
    "C:\Users\elmas\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Users\elmas\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Users\elmas\AppData\Local\Programs\Python\Python310\python.exe",
    "python",
    "python3"
)

foreach ($candidate in $candidates) {
    try {
        if (Test-Path $candidate -ErrorAction SilentlyContinue) {
            $ver = & $candidate -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')" 2>&1
            if ($LASTEXITCODE -eq 0 -and ($ver -match "Python 3\.12" -or $ver -match "Python 3\.11" -or $ver -match "Python 3\.10")) {
                $PYTHON = $candidate
                Write-Host "  OK Python compatible con PyTorch CUDA encontrado: $ver" -ForegroundColor Green
                Write-Host "     Ejecutable: $candidate" -ForegroundColor DarkGray
                break
            }
        } elseif ($candidate -notmatch "\\") {
            $ver = & $candidate -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')" 2>&1
            if ($LASTEXITCODE -eq 0 -and ($ver -match "Python 3\.12" -or $ver -match "Python 3\.11" -or $ver -match "Python 3\.10")) {
                $PYTHON = $candidate
                Write-Host "  OK Python compatible en PATH: $ver" -ForegroundColor Green
                break
            }
        }
    } catch { }
}

if (-not $PYTHON) {
    Write-Host "  ERROR: No se encontro Python 3.10-3.12." -ForegroundColor Red
    Write-Host "  PyTorch para Windows con soporte GPU (CUDA) requiere Python <= 3.12." -ForegroundColor Red
    Write-Host "  Descarga e instala Python 3.12.8 desde https://www.python.org/downloads/release/python-3128" -ForegroundColor Red
    exit 1
}

# ─────────────────────────────────────────────────────────────────────
# PASO 2: Detectar GPU NVIDIA y verificar soporte CUDA en PyTorch
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/5] Detectando GPU NVIDIA..." -ForegroundColor Yellow

$GPU_AVAILABLE = $false
$GPU_NAME = ""
$CUDA_IN_TORCH = $false

# Verificar si hay GPU NVIDIA con nvidia-smi
try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0 -and $nvidiaSmi -match "NVIDIA|GeForce|RTX|GTX|Quadro|Tesla") {
        $GPU_AVAILABLE = $true
        $GPU_NAME = $nvidiaSmi.Trim()
        Write-Host "  OK GPU NVIDIA detectada: $GPU_NAME" -ForegroundColor Green
    }
} catch {
    Write-Host "  INFO: nvidia-smi no encontrado en PATH" -ForegroundColor DarkGray
}

# Si nvidia-smi no esta en PATH, buscar directamente
if (-not $GPU_AVAILABLE) {
    $nvidiaPaths = @(
        "C:\Windows\System32\nvidia-smi.exe",
        "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    )
    foreach ($path in $nvidiaPaths) {
        if (Test-Path $path) {
            $nvidiaSmi = & $path --query-gpu=name --format=csv,noheader 2>&1
            if ($LASTEXITCODE -eq 0 -and $nvidiaSmi -match "NVIDIA|GeForce|RTX|GTX") {
                $GPU_AVAILABLE = $true
                $GPU_NAME = $nvidiaSmi.Trim()
                Write-Host "  OK GPU NVIDIA detectada: $GPU_NAME" -ForegroundColor Green
                break
            }
        }
    }
}

if (-not $GPU_AVAILABLE) {
    # Ultimo recurso: consultar WMI (no requiere drivers especiales)
    try {
        $gpuInfo = Get-WmiObject Win32_VideoController -ErrorAction SilentlyContinue |
                   Where-Object { $_.Name -match "NVIDIA|GeForce|RTX|GTX|Quadro" } |
                   Select-Object -First 1 -ExpandProperty Name
        if ($gpuInfo) {
            $GPU_AVAILABLE = $true
            $GPU_NAME = $gpuInfo
            Write-Host "  OK GPU NVIDIA detectada via WMI: $GPU_NAME" -ForegroundColor Green
        }
    } catch { }
}

if (-not $GPU_AVAILABLE) {
    Write-Host "  INFO: No se detecto GPU NVIDIA. Se usara CPU." -ForegroundColor DarkYellow
}

# Verificar si PyTorch tiene soporte CUDA
if ($GPU_AVAILABLE) {
    Write-Host "  Verificando soporte CUDA en PyTorch..." -ForegroundColor Cyan
    $cudaCheck = & $PYTHON -c "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NO')" 2>&1
    if ($cudaCheck -match "CUDA_OK") {
        $CUDA_IN_TORCH = $true
        $cudaVer = & $PYTHON -c "import torch; print(torch.version.cuda)" 2>&1
        $gpuTorchName = & $PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
        Write-Host "  OK PyTorch CUDA disponible (CUDA $cudaVer)" -ForegroundColor Green
        Write-Host "     GPU activa: $gpuTorchName" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "  AVISO: GPU detectada ($GPU_NAME) pero PyTorch NO tiene soporte CUDA." -ForegroundColor Yellow
        Write-Host "         La version actual de PyTorch fue instalada sin CUDA." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Reinstalando PyTorch con soporte CUDA 11.8 (compatible con RTX 2070)..." -ForegroundColor Cyan
        Write-Host "  (Esto puede tardar unos minutos - descarga ~2.5GB)" -ForegroundColor DarkGray
        Write-Host ""

        & $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

        if ($LASTEXITCODE -eq 0) {
            # Verificar de nuevo
            $cudaCheck2 = & $PYTHON -c "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NO')" 2>&1
            if ($cudaCheck2 -match "CUDA_OK") {
                $CUDA_IN_TORCH = $true
                $cudaVer2 = & $PYTHON -c "import torch; print(torch.version.cuda)" 2>&1
                Write-Host "  OK PyTorch CUDA reinstalado correctamente (CUDA $cudaVer2)" -ForegroundColor Green
            } else {
                Write-Host "  AVISO: La reinstalacion no activo CUDA. Verifica los drivers NVIDIA." -ForegroundColor Yellow
                Write-Host "         Descarga drivers desde: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
                Write-Host "         Continuando con CPU..." -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ERROR: Fallo la reinstalacion de PyTorch CUDA. Continuando con CPU." -ForegroundColor Red
        }
    }
}

# Resumen del device que se usara
Write-Host ""
if ($CUDA_IN_TORCH) {
    Write-Host "  >>> MODO: GPU (CUDA) - Reconstruccion acelerada por $GPU_NAME <<<" -ForegroundColor Magenta
} else {
    Write-Host "  >>> MODO: CPU - La reconstruccion sera lenta (~20 min/imagen) <<<" -ForegroundColor DarkYellow
}

# ─────────────────────────────────────────────────────────────────────
# PASO 3: Verificar e instalar dependencias faltantes
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/5] Verificando dependencias..." -ForegroundColor Yellow

$pkg_map = @{
    "numpy"       = "numpy"
    "torch"       = "torch"
    "torchvision" = "torchvision"
    "Pillow"      = "PIL"
    "tqdm"        = "tqdm"
    "scipy"       = "scipy"
    "omegaconf"   = "omegaconf"
    "ftfy"        = "ftfy"
    "regex"       = "regex"
    "einops"      = "einops"
    "requests"    = "requests"
}

$missing = @()
foreach ($pkg in $pkg_map.Keys) {
    $mod = $pkg_map[$pkg]
    $null = & $PYTHON -c "import $mod" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missing += $pkg
        Write-Host "  FALTA: $pkg" -ForegroundColor DarkYellow
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "  Instalando $($missing.Count) paquetes faltantes..." -ForegroundColor Cyan
    $req_file = Join-Path $SCRIPT_DIR "requirements_py312.txt"
    if (Test-Path $req_file) {
        & $PYTHON -m pip install -r $req_file
    } else {
        & $PYTHON -m pip install $missing
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK Dependencias instaladas." -ForegroundColor Green
    } else {
        Write-Host "  AVISO: Algunos paquetes fallaron. El script intentara continuar." -ForegroundColor Yellow
    }
} else {
    Write-Host "  OK Todas las dependencias criticas estan instaladas." -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────
# PASO 4: Verificar archivos del proyecto
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[4/5] Verificando archivos del proyecto..." -ForegroundColor Yellow

$required_files = @(
    "main_local_decoder.py",
    "config.py",
    "patch_taming.py",
    "pytorch_lightning_compat.py"
)

$missing_files = @()
foreach ($f in $required_files) {
    $path = Join-Path $SCRIPT_DIR $f
    if (Test-Path $path) {
        Write-Host "  OK $f" -ForegroundColor Green
    } else {
        Write-Host "  ERROR $f NO encontrado" -ForegroundColor Red
        $missing_files += $f
    }
}

$features_dir = Join-Path $SCRIPT_DIR "features"
if (Test-Path $features_dir) {
    Write-Host "  OK features/ existe" -ForegroundColor Green
} else {
    Write-Host "  AVISO features/ no encontrado - extrae features.tar.gz primero" -ForegroundColor Yellow
}

if ($missing_files.Count -gt 0) {
    Write-Host "  ERROR: Faltan archivos criticos. Verifica tu instalacion." -ForegroundColor Red
    exit 1
}

# ─────────────────────────────────────────────────────────────────────
# PASO 5: Ejecutar el pipeline
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[5/5] Iniciando pipeline de reconstruccion..." -ForegroundColor Yellow
if ($CUDA_IN_TORCH) {
    Write-Host "      Modo GPU activado - velocidad estimada: ~1-3 min/imagen" -ForegroundColor Green
} else {
    Write-Host "      Modo CPU - velocidad estimada: ~15-25 min/imagen" -ForegroundColor DarkYellow
}
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

$main_script = Join-Path $SCRIPT_DIR "main_local_decoder.py"

Push-Location $SCRIPT_DIR
try {
    & $PYTHON $main_script
    $exit_code = $LASTEXITCODE
} finally {
    Pop-Location
}

# ─────────────────────────────────────────────────────────────────────
# Resultado final
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan

if ($exit_code -eq 0) {
    Write-Host "  COMPLETADO exitosamente." -ForegroundColor Green
    Write-Host "  Imagenes en: $SCRIPT_DIR\output_reconstructions\" -ForegroundColor White
    if ($CUDA_IN_TORCH) {
        Write-Host "  (Procesado con GPU: $GPU_NAME)" -ForegroundColor Magenta
    }
} else {
    Write-Host "  El script termino con errores (codigo $exit_code)." -ForegroundColor Red
    Write-Host "  Log: $SCRIPT_DIR\output_reconstructions\reconstruction.log" -ForegroundColor Yellow
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
