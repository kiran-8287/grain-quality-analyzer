$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$env:TEMP = Join-Path $ProjectRoot 'tmp'
$env:TMP = $env:TEMP
$env:TORCH_HOME = Join-Path $ProjectRoot '.cache\torch'
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot '.cache\ultralytics'
$env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache\matplotlib'
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:TORCH_HOME,$env:YOLO_CONFIG_DIR,$env:MPLCONFIGDIR,$env:PIP_CACHE_DIR | Out-Null
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
& .\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
