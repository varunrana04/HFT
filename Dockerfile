FROM python:3.10-slim

WORKDIR /app

# Install necessary build tools for compiling the C++ engine
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project repository
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Compile the C++ Trading Engine (PyBind11 module)
RUN mkdir -p build && cd build && \
    cmake -DHFT_BUILD_PYTHON=ON -DHFT_BUILD_TESTS=OFF -DHFT_BUILD_BENCHMARKS=OFF .. && \
    make -j4

# Expose the dashboard/telemetry port
EXPOSE 8080

# Environment variables for the live engine
ENV HOST=0.0.0.0
ENV PORT=8080

# Run the live paper trading server
CMD ["python", "python/live_paper_trade.py"]
