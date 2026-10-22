# quant.ps1 — One-command launcher for the quant research environment
# Usage: .\quant.ps1 [optional script to run]
#
# Examples:
#   .\quant.ps1                          # interactive Python 3.11 shell
#   .\quant.ps1 scratch\train_weights.py # run a script
#   .\quant.ps1 -c "import hftbacktest"  # run inline code

param(
    [string]$Script = "",
    [string]$Command = ""
)

$PYTHON = "$PSScriptRoot\quant_env\Scripts\python.exe"

if (-not (Test-Path $PYTHON)) {
    Write-Error "quant_env not found. Run: py -3.11 -m venv quant_env && .\quant_env\Scripts\python.exe -m pip install hftbacktest vectorbt numba ccxt lightgbm scikit-learn"
    exit 1
}

Write-Host "[quant] Python: $(&$PYTHON --version)" -ForegroundColor Cyan
Write-Host "[quant] Packages: hftbacktest, vectorbt, numba, ccxt, lightgbm" -ForegroundColor Cyan

if ($Command) {
    & $PYTHON -c $Command
} elseif ($Script) {
    & $PYTHON $Script
} else {
    Write-Host "[quant] Entering interactive Python 3.11 shell. Type exit() to quit." -ForegroundColor Yellow
    & $PYTHON
}
