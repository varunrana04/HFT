@echo off
REM ════════════════════════════════════════════════════
REM   HFT Engine — Build, Test, Benchmark Script
REM   Run this from the HFT project root directory
REM ════════════════════════════════════════════════════

echo.
echo ====================================================
echo   HFT Engine — Full Build Pipeline
echo ====================================================
echo.

REM ── Step 1: CMake Configure ──
echo [1/4] Configuring CMake...
set PATH=%PATH%;%APPDATA%\Python\Python314\Scripts;%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configure failed!
    echo Make sure CMake 3.20+ is installed and in PATH.
    pause
    exit /b 1
)
echo [OK] CMake configured successfully.
echo.

REM ── Step 2: Build ──
echo [2/4] Building project...
cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed! Check compiler errors above.
    pause
    exit /b 1
)
echo [OK] Build successful.
echo.

REM ── Step 3: Run Tests ──
echo [3/4] Running unit tests (36+ tests)...
cd build
ctest --output-on-failure -C Release
set TEST_RESULT=%ERRORLEVEL%
cd ..
if %TEST_RESULT% neq 0 (
    echo [WARNING] Some tests failed. See output above.
) else (
    echo [OK] All tests passed!
)
echo.

REM ── Step 4: Run Benchmarks ──
echo [4/4] Running latency benchmarks...
if exist "build\Release\hft_bench.exe" (
    build\Release\hft_bench.exe
) else if exist "build\hft_bench.exe" (
    build\hft_bench.exe
) else (
    echo [WARNING] hft_bench executable not found. Build may have issues.
)
echo.

echo ====================================================
echo   Build Pipeline Complete!
echo ====================================================
echo.
echo Next steps:
echo   1. python python/train_model.py --synthetic
echo   2. python python/backtest.py --data data/your_data.csv
echo   3. python python/mt5_gateway.py --symbol EURUSD
echo.
pause
