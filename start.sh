#!/bin/bash
set -e

# Validate required environment variables before starting anything
if [ -z "$DELTA_API_KEY" ] || [ -z "$DELTA_API_SECRET" ]; then
    echo "============================================"
    echo "ERROR: DELTA_API_KEY and DELTA_API_SECRET"
    echo "must be set in Render > Environment."
    echo "Go to: dashboard.render.com → your service"
    echo "       → Environment → Add env vars"
    echo "============================================"
    exit 1
fi

echo "Starting C++ HFT Engine on port 8081..."
./hft_engine_live &
ENGINE_PID=$!

# Give engine 3 seconds to start, then check it's still alive
sleep 3
if ! kill -0 $ENGINE_PID 2>/dev/null; then
    echo "ERROR: HFT engine exited immediately. Check logs above."
    exit 1
fi

echo "Starting NGINX on port 8080..."
nginx -g 'daemon off;'
