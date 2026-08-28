FROM python:3.11-slim-bookworm

WORKDIR /app

# ── System deps ────────────────────────────────────────────────────────────────
# Only the bare minimum needed for the Python-only path.
# The C++ engine (hft_engine.pyd) is an optional enhancement; if cmake fails
# the service falls back automatically to pure_python_engine.py.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    git \
    curl \
    build-essential \
    ca-certificates \
    libssl-dev \
    libsimdjson-dev \
    && rm -rf /var/lib/apt/lists/* \
    # Download simdjson amalgamation to the path CMakeLists.txt expects
    && mkdir -p /usr/local/include/simdjson \
    && curl -fsSL https://raw.githubusercontent.com/simdjson/simdjson/v3.10.1/singleheader/simdjson.h \
         -o /usr/local/include/simdjson/simdjson.h \
    && curl -fsSL https://raw.githubusercontent.com/simdjson/simdjson/v3.10.1/singleheader/simdjson.cpp \
         -o /usr/local/include/simdjson/simdjson.cpp

# ── Python deps ────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# ── Source ─────────────────────────────────────────────────────────────────────
COPY . .

# ── Optional: build C++ PyBind11 engine ────────────────────────────────────────
# This is best-effort. If cmake or the build fails (e.g. missing simdjson header)
# the Python process falls back to pure_python_engine.py automatically via
# engine_loader.py, so the service still starts correctly.
RUN mkdir -p build && cd build && \
    cmake \
      -DHFT_BUILD_PYTHON=ON \
      -DHFT_BUILD_TESTS=OFF \
      -DHFT_BUILD_BENCHMARKS=OFF \
      -DHFT_USE_DPDK=OFF \
      -DUSE_AVX512=OFF \
      .. && \
    make -j$(nproc) 2>&1 || \
    echo "[DOCKER] C++ build failed -> pure_python_engine.py fallback will be used" ; \
    # Copy built .so next to engine_loader.py so it is found on the Python path
    find build -name 'hft_engine*.so' -exec cp {} python/ \; 2>/dev/null || true

# ── Port ────────────────────────────────────────────────────────────────────────
EXPOSE 8080

ENV HOST=0.0.0.0
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
CMD ["python", "python/live_paper_trade.py"]
