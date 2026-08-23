# Use Ubuntu 22.04 LTS as the base image for stability and performance
FROM ubuntu:22.04

# Avoid tzdata prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, C++ toolchain, and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    ninja-build \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Set up Python virtual environment
RUN python3.11 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
# Ensure pybind11 is installed for CMake to find it
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt pybind11

# Build the C++ core and Pybind11 extension via CMake
# Using Ninja for faster builds and better parallelism
RUN mkdir -p build && cd build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYBIND11_DIR=$(python -m pybind11 --cmakedir) .. && \
    ninja && \
    cp hft_engine*.so ../python/ || echo "Build output already in python directory"

# Default command: Run the live paper trading / ML bridge
CMD ["python", "python/live_paper_trade.py"]
