$ErrorActionPreference = 'Stop'
$PipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $PipelineRoot
Set-Location $PipelineRoot
$WorkRoot = Join-Path $RepoRoot 'data\rice_count\work'
$env:TEMP = Join-Path $WorkRoot 'tmp'
$env:TMP = $env:TEMP
$CacheRoot = Join-Path $WorkRoot 'runtime_cache'
$env:TORCH_HOME = Join-Path $CacheRoot 'torch'
$env:YOLO_CONFIG_DIR = Join-Path $CacheRoot 'ultralytics'
$env:MPLCONFIGDIR = Join-Path $CacheRoot 'matplotlib'
$env:PIP_CACHE_DIR = Join-Path $CacheRoot 'pip'
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:TORCH_HOME,$env:YOLO_CONFIG_DIR,$env:MPLCONFIGDIR,$env:PIP_CACHE_DIR | Out-Null
$Python = Join-Path $RepoRoot 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $Environment = Split-Path -Parent (Split-Path -Parent $Python)
    py -3.11 -m venv $Environment
}
$PythonVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($PythonVersion -ne '3.11') {
    throw "The shared backend environment must use Python 3.11 (found $PythonVersion)."
}
& $Python -m pip install --no-cache-dir --upgrade pip
& $Python -m pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
& $Python -m pip install --no-cache-dir -r (Join-Path $RepoRoot 'backend\requirements.txt')
& $Python -m pip install --no-cache-dir -r (Join-Path $PipelineRoot 'requirements.txt')
& $Python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
