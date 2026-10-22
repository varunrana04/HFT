# Institutional Model Training & Calibration Guide

This guide details the complete suite of quantitative machine learning model training pipelines in the HFT engine.

---

## 🏗 Model Architecture Overview

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              L2 Market Data & Trade Tape                │
                        │        (bookTicker + aggTrades @ Binance Futures)       │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┴─────────────────────────────────┐
                 │                                                                    │
                 ▼                                                                    ▼
┌─────────────────────────────────┐                               ┌─────────────────────────────────┐
│     LightGBM Signal Booster     │                               │    Gaussian HMM Regime Model    │
│  • 8 L2 Microstructure Signals  │                               │  • 4 Continuous Regimes         │
│  • Asymmetric Adverse Loss      │                               │  • P(Regime | Vol, Spread)      │
│  • Fast Flat-Array (0.23 µs)    │                               │  • Dynamic Quote Expansion      │
└────────────────┬────────────────┘                               └────────────────┬────────────────┘
                 │                                                                 │
                 └──────────────────────────────────┬──────────────────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │      Online RL Policy Adapter     │
                                  │  • Continuous Actor-Critic PPO    │
                                  │  • Fee-Aware Continuous Reward    │
                                  │  • Reservation Skew & Half-Spread │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │   Almgren-Chriss Execution Engine │
                                  │  • Hyperbolic sinh(κt) Trajectory │
                                  │  • Pre-Trade Risk Budgeting Gates │
                                  └───────────────────────────────────┘
```

---

## ⚡ Quickstart: Central Training CLI

All models can be calibrated or re-trained via the central CLI:

```bash
# Train ALL models in sequence (HMM -> LightGBM -> Ridge -> Online RL)
python python/train_all_models.py --model all

# Train specific models
python python/train_all_models.py --model lgbm --rounds 150
python python/train_all_models.py --model hmm
python python/train_all_models.py --model ridge
python python/train_all_models.py --model rl
```

---

## 📁 Model Training Files Directory

| Model File | Script Path | Output Artifact | Description |
| :--- | :--- | :--- | :--- |
| **All Models** | [`python/train_all_models.py`](file:///c:/Users/Varun/Downloads/HFT/python/train_all_models.py) | All artifacts below | Unified orchestrator for multi-model training & validation. |
| **LightGBM (Local)** | [`python/ml_pipeline/train_lightgbm.py`](file:///c:/Users/Varun/Downloads/HFT/python/ml_pipeline/train_lightgbm.py) | `models/lgbm_signal_model.txt` | Trains 8-feature microstructure GBDT with asymmetric penalty. |
| **LightGBM (Kaggle)**| [`python/kaggle/train_lgbm_kaggle.py`](file:///c:/Users/Varun/Downloads/HFT/python/kaggle/train_lgbm_kaggle.py) | `lgbm_trade_tape_model.txt` | Ingests 2.5 years of Binance Vision data chunk-by-chunk in memory. |
| **HMM Regime** | [`python/ml_pipeline/train_hmm_regime.py`](file:///c:/Users/Varun/Downloads/HFT/python/ml_pipeline/train_hmm_regime.py) | `models/hmm_regime.pkl` | 4-state Gaussian mixture on realized volatility & spread. |
| **Online RL (Replay)**| [`scripts/train_online_rl_offline.py`](file:///c:/Users/Varun/Downloads/HFT/scripts/train_online_rl_offline.py) | `models/online_rl_weights.json` | Calibrates online RL weights across recorded live fills & ticks. |
| **PPO RL (Kaggle)** | [`python/kaggle/train_rl_ppo.py`](file:///c:/Users/Varun/Downloads/HFT/python/kaggle/train_rl_ppo.py) | `models/rl_policy.pt` | Continuous PPO agent learning dynamic spread multiplier & gamma. |
| **Ridge Combiner** | [`python/ml_pipeline/train_model.py`](file:///c:/Users/Varun/Downloads/HFT/python/ml_pipeline/train_model.py) | `models/signal_weights.bin` | Walk-forward cross-validated 96-byte binary linear combiner. |
| **Live Ridge** | [`scripts/train_ridge_live.py`](file:///c:/Users/Varun/Downloads/HFT/scripts/train_ridge_live.py) | `models/signal_weights.bin` | Online recursive ridge regression streaming trainer. |

---

## 1. LightGBM Microstructure Signal Model (8 Features)

### Feature Schema
The model operates on 8 high-frequency L2 signals:
1. `tfi_ema_50`: Trade flow imbalance fast EMA (aggTrades flow)
2. `tfi_ema_200`: Trade flow imbalance slow EMA
3. `volatility_ema_100`: High-frequency realized volatility fast EMA
4. `volatility_ema_500`: High-frequency realized volatility slow EMA
5. `book_imbalance`: Cont-Kukanov Order Book Imbalance $(Q_{bid} - Q_{ask}) / (Q_{bid} + Q_{ask})$
6. `spread_bps`: Real Level-2 bid/ask spread $(P_{ask} - P_{bid}) / P_{mid} \times 10,000$
7. `microprice`: Volume-weighted fair price $(P_{bid} Q_{ask} + P_{ask} Q_{bid}) / (Q_{bid} + Q_{ask})$
8. `spread_ema`: 50-period EMA of spread in bps

### Custom Asymmetric Loss Formulation
In high-frequency market making, a false positive (predicting a price move that fails to materialize) incurs adverse selection and crossing fees, whereas a false negative only incurs opportunity cost.
$$\mathcal{L}(y, \hat{y}) = \begin{cases} 1.8 \cdot (y - \hat{y})^2 & \text{if } |\hat{y}| > 0.5 \text{ and } |y| \le 0.5 \text{ (False Positive)} \\ 1.0 \cdot (y - \hat{y})^2 & \text{otherwise} \end{cases}$$

### Local Training
```bash
python python/ml_pipeline/train_lightgbm.py --rounds 150
```

---

## 2. Gaussian HMM Regime Model

Classifies instantaneous microstructure into 4 market regimes:
1. **Regime 0 (MEAN_REVERTING)**: Volatility < 0.8, Spread < 0.6 bps (Quote multiplier: 0.9x)
2. **Regime 1 (MOMENTUM_TREND)**: Volatility 1.0 - 1.4, Spread 0.7 - 1.1 bps (Quote multiplier: 1.1x)
3. **Regime 2 (HIGH_VOL_WIDE)**: Volatility 1.8 - 2.6, Spread 1.4 - 2.0 bps (Quote multiplier: 1.5x)
4. **Regime 3 (TOXIC_CRISIS)**: Volatility > 3.5, Spread > 2.8 bps (Quote multiplier: 2.4x / Risk clamp)

### Training Command
```bash
python python/ml_pipeline/train_hmm_regime.py
```

---

## 3. Online RL Policy Calibration (Offline Replay)

Ingests recorded live orderbook ticks and fill events from `data/live/` to perform policy gradient updates:
- **State Vector (13 dims)**: OFI, VPIN, Buyer Aggression, Seller Aggression, Book Collapse rate, Microprice bias, Spread ratio, Options Delta, LLM priors, and inventory.
- **Reward**: Net realized PnL minus maker exchange fees ($-0.5$ bps maker rebate accounted) minus inventory penalty.
- **Output**: `models/online_rl_weights.json` for live low-latency linear forward passes.

### Training Command
```bash
python scripts/train_online_rl_offline.py
```

---

## 4. Kaggle Cloud GPU Training (2.5 Years Ingestion)

For massive multi-year dataset training:
1. Open Kaggle notebook and set accelerator to **GPU T4 x2**.
2. Paste the contents of [`python/kaggle/train_lgbm_kaggle.py`](file:///c:/Users/Varun/Downloads/HFT/python/kaggle/train_lgbm_kaggle.py) for LightGBM or [`python/kaggle/train_rl_ppo.py`](file:///c:/Users/Varun/Downloads/HFT/python/kaggle/train_rl_ppo.py) for PPO RL.
3. The scripts automatically stream raw daily ZIP archives from `https://data.binance.vision`, construct the 8-feature dataset in Polars, and export model weights to `/kaggle/working/models/`.
4. Download the generated `lgbm_trade_tape_model.txt` or `rl_policy.pt` and place into `models/`.
