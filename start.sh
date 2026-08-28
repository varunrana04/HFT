#!/bin/bash
set -e

echo "Starting C++ HFT Engine on port 8081..."
./hft_engine_live &

echo "Starting NGINX on port 8080..."
nginx -g 'daemon off;'
