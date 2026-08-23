$ErrorActionPreference = "Stop"

$workspace = (Get-Location).Path
$cmakeDir = "$workspace\toolchain\cmake-3.29.3-windows-x86_64\bin"
$mingwDir = "$workspace\toolchain\mingw64\bin"

# Add toolchains to PATH temporarily
$env:PATH = "$cmakeDir;$mingwDir;" + $env:PATH

# Attempt to use the uv virtual environment first, fallback to system python
if ($env:PYTHON_EXE) {
    $pythonExe = $env:PYTHON_EXE
} else {
    $pythonExe = "$workspace\.venv\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        $pythonExe = "python.exe"
    }
}

Write-Host "Building HFT Engine C++ Extension using Python: $pythonExe..." -ForegroundColor Cyan

# Create build directory
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Force -Path "build" | Out-Null
}

# Run CMake Configuration
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$pythonExe" -DPYTHON_EXECUTABLE="$pythonExe"

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed!"
    exit 1
}

# Run CMake Build
cmake --build build --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake build failed!"
    exit 1
}

Write-Host "Build Successful!" -ForegroundColor Green
