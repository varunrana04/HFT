# setup_and_build.ps1
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "  HFT Engine - Portable Compiler Setup and Build" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

$MingwDir = $env:USERPROFILE + "\mingw64"
$ZipPath = $env:USERPROFILE + "\mingw.zip"
$Url = "https://github.com/brechtsanders/winlibs_mingw/releases/download/14.2.0posix-18.1.8-12.0.0-msvcrt-r1/winlibs-x86_64-posix-seh-gcc-14.2.0-mingw-w64msvcrt-12.0.0-r1.zip"

# Step 1: Download Compiler
if (-Not (Test-Path ($MingwDir + "\bin\g++.exe"))) {
    Write-Host "[1/4] Downloading Portable C++ Compiler (GCC 14)... this will take a minute or two!" -ForegroundColor Yellow
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath
    
    Write-Host "[2/4] Extracting compiler... (please wait, this can take 2-3 minutes)" -ForegroundColor Yellow
    Expand-Archive -Path $ZipPath -DestinationPath $env:USERPROFILE -Force
    Remove-Item $ZipPath
} else {
    Write-Host "[1/4] Portable C++ Compiler already exists! Skipping download." -ForegroundColor Green
}

# Step 2: Setup Environment
Write-Host "[3/4] Configuring environment..." -ForegroundColor Yellow
$env:PATH = $MingwDir + "\bin;" + $env:PATH
$env:PATH = $env:APPDATA + "\Python\Python314\Scripts;" + $env:PATH
$env:PATH = $env:USERPROFILE + "\AppData\Roaming\Python\Python314\Scripts;" + $env:PATH

# Step 3: Clean and Build
Write-Host "[4/4] Compiling HFT Engine..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure failed!" -ForegroundColor Red
    exit 1
}

cmake --build build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "====================================================" -ForegroundColor Green
Write-Host "  BUILD SUCCESSFUL! You can now run the benchmarks:" -ForegroundColor Green
Write-Host "  .\build\hft_bench.exe" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
