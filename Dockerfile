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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the CMake project files
COPY CMakeLists.txt .
COPY cpp/ cpp/

# Build the C++ engine
RUN mkdir build && cd build && cmake .. && make -j$(nproc) hft_engine_live

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

# Copy entrypoint script
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
