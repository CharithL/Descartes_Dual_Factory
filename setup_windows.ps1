# L5PC DESCARTES — Windows Setup Script
#
# Prerequisites:
#   - Anaconda or Miniconda installed
#   - Git installed
#   - NVIDIA GPU with CUDA drivers (RTX 5080, etc.)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  L5PC DESCARTES — Windows Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check for conda
$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($null -eq $conda) {
    Write-Host "ERROR: conda not found in PATH." -ForegroundColor Red
    Write-Host "Please install Miniconda from:" -ForegroundColor Yellow
    Write-Host "  https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor White
    Write-Host ""
    Write-Host "After installing, restart this terminal and re-run this script."
    exit 1
}

Write-Host "[1/5] Creating conda environment 'l5pc'..." -ForegroundColor Green

# Check if environment already exists
$envExists = conda env list | Select-String "l5pc"
if ($envExists) {
    Write-Host "  Environment 'l5pc' already exists. Activating..." -ForegroundColor Yellow
} else {
    conda create -n l5pc python=3.11 -y
}

Write-Host ""
Write-Host "[2/5] Installing NEURON via conda-forge..." -ForegroundColor Green
conda install -n l5pc -c conda-forge neuron -y

Write-Host ""
Write-Host "[3/5] Installing PyTorch with CUDA support..." -ForegroundColor Green
# PyTorch with CUDA 12.4 (RTX 5080 compatible)
conda run -n l5pc pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
Write-Host "[4/5] Installing remaining Python dependencies..." -ForegroundColor Green
conda run -n l5pc pip install -r requirements-windows.txt

Write-Host ""
Write-Host "[5/5] Downloading and compiling NEURON mechanisms..." -ForegroundColor Green
conda run -n l5pc powershell -ExecutionPolicy Bypass -File scripts\download_mechanisms.ps1

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor Cyan
Write-Host "  conda activate l5pc" -ForegroundColor White
Write-Host ""
Write-Host "To run the pipeline:" -ForegroundColor Cyan
Write-Host "  python scripts\run_phase1.py" -ForegroundColor White
Write-Host ""
Write-Host "To run just the training (if you have simulation data):" -ForegroundColor Cyan
Write-Host "  python scripts\run_phase1.py --start-step 2" -ForegroundColor White
Write-Host ""
Write-Host "GPU check:" -ForegroundColor Cyan
Write-Host "  conda activate l5pc" -ForegroundColor White
Write-Host "  python -c `"import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')`"" -ForegroundColor White
