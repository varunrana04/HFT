$env:DELTA_API_KEY="your_api_key_here"
$env:DELTA_API_SECRET="your_api_secret_here"

# Add MSYS2 paths so the executable can find libcrypto.dll and other MinGW DLLs
$env:PATH += ";C:\msys64\mingw64\bin;C:\msys64\usr\bin"

Write-Host "Starting HFT Engine..." -ForegroundColor Green
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptPath
cd build
./hft_engine_live.exe
