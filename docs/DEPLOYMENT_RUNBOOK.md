# HFT Engine Deployment Runbook

## Overview
This runbook details the procedures for deploying the HFT engine into a live production environment. The architecture is a hybrid C++/Python system designed for portability and analytical flexibility, running efficiently on standard cloud compute instances.

## Hardware Requirements
- **Location**: AWS Tokyo (ap-northeast-1) for Binance Futures.
- **Instance Type**: Compute-optimized (e.g., c6i.large or higher).
- **Networking**: Enhanced Networking (ENA) enabled.

## Architecture
The system uses a highly optimized C++ core (`StrategyEngine`) wrapped in Python via pybind11 (`engine_loader`). 
Python handles websocket ingestion, telemetry, and machine learning model updates (True HMM + StatArb).
C++ handles order book construction, feature extraction, signal combination, and risk management.

## Deployment Instructions

### 1. Environment Setup
Create a `.env` file containing your production API credentials based on `.env.example`:
```bash
BINANCE_API_KEY=your_production_api_key_here
BINANCE_API_SECRET=your_production_api_secret_here
```

### 2. Docker Execution
Build the container:
```bash
docker build -t hft_engine:latest .
```

Run the container:
```bash
docker run -d \
  --name hft_live_node \
  --network host \
  --env-file .env \
  hft_engine:latest
```

## Risk Management (Post-Deployment)
The engine has been upgraded with critical safeguards for production:
1. **Maximum Inventory Hard Limit**: If net absolute position exceeds the set limit, the `RiskManager` instantly rejects all new `LIMIT` (Maker) signals in that direction.
2. **Stale Quote Protection**: Orders resting in the book for > 2.5 seconds are flagged as stale and cancelled to prevent adverse selection (sniping).
3. **Session Lock**: The `live_paper_trade.py` orchestrator utilizes a strict PID-aware lockfile to prevent multi-writer journal corruption.
