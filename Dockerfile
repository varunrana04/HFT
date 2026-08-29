# ── Build Stage ──────────────────────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    cmake \
    make \
    libssl-dev \
    ca-certificates \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the CMake project files
COPY CMakeLists.txt .
# Bust the cache to ensure Render rebuilds the engine with the Ping fix
ENV CACHE_BUSTER=1
COPY cpp/ cpp/

# Download simdjson single-header
RUN mkdir -p /usr/local/include/simdjson && \
    wget -O /usr/local/include/simdjson/simdjson.h https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.h && \
    wget -O /usr/local/include/simdjson/simdjson.cpp https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.cpp

# Build the C++ engine (disable benchmarks/python bindings, -j1 for OOM safety)
RUN mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DHFT_BUILD_BENCHMARKS=OFF \
      -DHFT_BUILD_PYTHON=OFF \
      -DHFT_BUILD_TESTS=OFF && \
    make -j1 hft_engine_live

# ── Runtime Stage ────────────────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    ca-certificates \
    nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled engine from the builder stage
COPY --from=builder /app/build/hft_engine_live .

# Setup NGINX config
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the dashboard files to NGINX web root
COPY live_dashboard/ /var/www/html/
RUN chmod -R 777 /var/www/html/

# Copy entrypoint script
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
