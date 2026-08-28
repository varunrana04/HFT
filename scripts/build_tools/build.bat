@echo off
echo ======================================
echo  HFT Engine - Build Script (Windows)
echo ======================================

IF NOT EXIST ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install pybind11

echo [INFO] Building C++ Core ^& Pybind11 Extension...
IF NOT EXIST "build" mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja

echo [INFO] Copying python bridge extension...
copy hft_engine*.pyd ..\python\ >nul 2>&1
cd ..

echo [SUCCESS] Build complete! Run 'python python\live_paper_trade.py' to start.
