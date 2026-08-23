#!/bin/bash
set -e

echo "======================================"
echo " HFT Engine - Build Script (Linux/Mac) "
echo "======================================"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt pybind11

echo "[INFO] Building C++ Core & Pybind11 Extension..."
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYBIND11_DIR=$(python -m pybind11 --cmakedir) ..
ninja

echo "[INFO] Copying python bridge extension..."
cp hft_engine*.so ../python/ || echo "Build output already in python directory"

echo "[SUCCESS] Build complete! Run 'python python/live_paper_trade.py' to start."
