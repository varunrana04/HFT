import sys, struct, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

df = pd.read_csv("data/btcusdt_1m.csv", header=0)
df.columns = ["open_time","open","high","low","close","volume","close_time",
              "quote_vol","trades","taker_buy_vol","taker_buy_quote_vol","ignore"]
df = df.astype({"close": float, "volume": float, "taker_buy_vol": float,
                "high": float, "low": float})

close = df["close"]
vol   = df["volume"]
tbv   = df["taker_buy_vol"]

obi       = (tbv / vol.clip(lower=1e-9)) * 2 - 1
log_ret   = np.log(close / close.shift(1))
realized_vol = log_ret.rolling(20).std().fillna(0)
ofi       = (tbv - (vol - tbv)).rolling(5).sum().fillna(0)
ofi_norm  = ofi / (vol.rolling(5).sum().clip(lower=1e-9) * close)
micro_ret = log_ret.fillna(0)
spread_bps = ((df["high"] - df["low"]) / close * 1e4)
cvd_raw   = (tbv - (vol - tbv)).rolling(30).sum().fillna(0)
vpin_raw  = abs(tbv / vol.clip(lower=1e-9) - 0.5) * 2
vpin      = vpin_raw.rolling(50).mean().fillna(0.5)

X = pd.DataFrame({
    "w_obi":        obi,
    "w_vpin":       vpin,
    "w_vol":        realized_vol,
    "w_spread":     -spread_bps / 10,
    "w_ofi":        ofi_norm,
    "w_microprice": micro_ret * 1e3,
    "w_bias":       1.0,
}).dropna()

fwd_ret = log_ret.shift(-5).reindex(X.index).dropna()
X = X.loc[fwd_ret.index]

split = int(len(X) * 0.70)
Xtr, Xte = X.iloc[:split], X.iloc[split:]
ytr, yte = fwd_ret.iloc[:split], fwd_ret.iloc[split:]

model = Ridge(alpha=1.0)
model.fit(Xtr, ytr)
oos_r2 = model.score(Xte, yte)
print(f"OOS R²: {oos_r2:.6f}   (n_train={split:,}, n_val={len(Xte):,})")

weights = dict(zip(X.columns, model.coef_))
weights["w_bias"] = float(model.intercept_)
print("Weights:", {k: f"{v:.6f}" for k,v in weights.items()})

KEY_ORDER = ["w_obi","w_vpin","w_vol","w_spread","w_ofi","w_microprice","w_bias"]
vals = [weights.get(k, 0.0) for k in KEY_ORDER]

os.makedirs("models", exist_ok=True)
with open("models/signal_weights.bin", "wb") as f:
    f.write(struct.pack(f"{len(vals)}d", *vals))

import json
with open("models/signal_weights_meta.json", "w") as f:
    json.dump({"keys": KEY_ORDER, "values": vals,
               "oos_r2": oos_r2, "n_train": split, "n_val": len(Xte)}, f, indent=2)

print("Written: models/signal_weights.bin (56 bytes) + models/signal_weights_meta.json")
