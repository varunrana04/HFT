FROM python:3.10-slim-bookworm

WORKDIR /app

# Install necessary build tools for compiling the C++ engine
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    build-essential \
    libdpdk-dev \
    libnuma-dev \
    wget \
    zlib1g-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uWebSockets & uSockets
RUN git clone --recursive https://github.com/uNetworking/uWebSockets.git /tmp/uWebSockets && \
    cd /tmp/uWebSockets/uSockets && make && \
    cp *.a /usr/local/lib/ && cp src/*.h /usr/local/include/ && \
    cd /tmp/uWebSockets && cp -r src/* /usr/local/include/ && \
    rm -rf /tmp/uWebSockets

# Install simdjson (single-header)
RUN mkdir -p /usr/local/include/simdjson && \
    wget https://cdn.jsdelivr.net/gh/simdjson/simdjson@master/singleheader/simdjson.h -O /usr/local/include/simdjson/simdjson.h && \
    wget https://cdn.jsdelivr.net/gh/simdjson/simdjson@master/singleheader/simdjson.cpp -O /usr/local/include/simdjson/simdjson.cpp

# Download and install ONNX Runtime C++ API
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.1/onnxruntime-linux-x64-1.16.1.tgz && \
    tar -xzf onnxruntime-linux-x64-1.16.1.tgz && \
    cp -r onnxruntime-linux-x64-1.16.1/include/* /usr/local/include/ && \
    cp -r onnxruntime-linux-x64-1.16.1/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm -rf onnxruntime-linux-x64-1.16.1*

# Copy requirements first to cache the slow pip install step
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the entire project repository
COPY . .

# Compile the C++ Trading Engine (PyBind11 module) without DPDK and AVX-512 for laptop compatibility
RUN mkdir -p build && cd build && \
    cmake -DHFT_BUILD_PYTHON=ON -DHFT_BUILD_TESTS=OFF -DHFT_BUILD_BENCHMARKS=OFF -DHFT_USE_DPDK=OFF -DUSE_AVX512=OFF .. && \
    make VERBOSE=1 -j$(nproc)

# Expose the dashboard/telemetry port
EXPOSE 8080

# Environment variables for the live engine
ENV HOST=0.0.0.0
ENV PORT=8080

# Run the live paper trading server
CMD ["python", "python/live_paper_trade.py"]
