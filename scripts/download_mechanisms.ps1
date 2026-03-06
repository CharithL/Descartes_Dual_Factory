# Download BBP ion channel mechanism .mod files and compile with nrnivmodl.
# These are the Hay et al. 2011 / Bahl et al. 2012 mechanisms used by the L5PC model.
#
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts\download_mechanisms.ps1
#
# Sources:
#   1. BlueBrain/BluePyOpt GitHub repo (fast, reliable)

$ErrorActionPreference = "Stop"

# Navigate to project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$ModDir = "mechanisms"
if (-not (Test-Path $ModDir)) {
    New-Item -ItemType Directory -Path $ModDir | Out-Null
}

Write-Host "=== Downloading BBP ion channel mechanisms ===" -ForegroundColor Cyan

# List of mechanism files needed
$Mechanisms = @(
    "Ca_HVA.mod",
    "Ca_LVAst.mod",
    "CaDynamics_E2.mod",
    "Ih.mod",
    "Im.mod",
    "K_Pst.mod",
    "K_Tst.mod",
    "Nap_Et2.mod",
    "NaTa_t.mod",
    "SK_E2.mod",
    "SKv3_1.mod"
)

$GithubBase = "https://raw.githubusercontent.com/BlueBrain/BluePyOpt/master/examples/l5pc/mechanisms"

$Downloaded = 0
foreach ($ModFile in $Mechanisms) {
    $OutPath = Join-Path $ModDir $ModFile
    if (Test-Path $OutPath) {
        Write-Host "  [exists] $ModFile" -ForegroundColor Green
        $Downloaded++
        continue
    }

    Write-Host -NoNewline "  Downloading $ModFile ... "
    $Url = "$GithubBase/$ModFile"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutPath -TimeoutSec 30 -ErrorAction Stop
        Write-Host "OK" -ForegroundColor Green
        $Downloaded++
    }
    catch {
        Write-Host "FAILED" -ForegroundColor Red
        if (Test-Path $OutPath) { Remove-Item $OutPath }
    }
}

Write-Host ""
Write-Host "Downloaded $Downloaded/$($Mechanisms.Count) mechanism files."

if ($Downloaded -lt $Mechanisms.Count) {
    Write-Host "WARNING: Some mechanisms failed to download." -ForegroundColor Yellow
    Write-Host "Check your internet connection and try again."
}

Write-Host ""
Write-Host "=== Compiling mechanisms with nrnivmodl ===" -ForegroundColor Cyan

# Check if nrnivmodl is available
$nrnivmodl = Get-Command nrnivmodl -ErrorAction SilentlyContinue
if ($null -eq $nrnivmodl) {
    Write-Host "WARNING: nrnivmodl not found in PATH." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To compile mechanisms, you need NEURON installed." -ForegroundColor Yellow
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  1. conda install -c conda-forge neuron   (recommended for Windows)" -ForegroundColor White
    Write-Host "  2. Use WSL2: wsl bash scripts/download_mechanisms.sh" -ForegroundColor White
    Write-Host ""
    Write-Host "After installing NEURON, run from the project root:" -ForegroundColor Yellow
    Write-Host "  cd mechanisms && nrnivmodl . && cd .." -ForegroundColor White
    exit 1
}

Push-Location $ModDir
try {
    nrnivmodl .
    Write-Host ""
    Write-Host "=== Done ===" -ForegroundColor Green
    Write-Host "Compiled mechanisms are in: $ModDir/" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: nrnivmodl compilation failed." -ForegroundColor Red
    Write-Host "Make sure NEURON is properly installed with:" -ForegroundColor Yellow
    Write-Host "  conda install -c conda-forge neuron" -ForegroundColor White
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Now run: python scripts\run_phase1.py" -ForegroundColor Cyan
