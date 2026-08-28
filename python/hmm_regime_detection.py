import pandas as pd
import numpy as np
import argparse
import joblib
import os

def load_data(filepath: str, max_rows: int = 500_000) -> pd.DataFrame:
    print(f"Loading {max_rows} rows from {filepath} for HMM training...")
    df = pd.read_csv(filepath, nrows=max_rows)
    if 'timestamp_ns' in df.columns:
        df = df.sort_values('timestamp_ns').reset_index(drop=True)
    return df

def train_hmm(df: pd.DataFrame, n_components: int = 3):
    """Train a True Hidden Markov Model (HMM) on volatility and spread."""
    print("Preparing features for True HMM (Regime Detection)...")
    
    features = []
    if 'realized_vol' in df.columns:
        features.append(df['realized_vol'].values)
    elif 'mid_price' in df.columns:
        vol = pd.Series(df['mid_price']).pct_change().rolling(100).std().fillna(0).values
        features.append(vol)
        
    if 'spread_bps' in df.columns:
        features.append(df['spread_bps'].values)
    else:
        features.append(np.ones(len(df)))
        
    if len(features) == 0:
        print("[ERROR] No suitable features for Regime Detection.")
        return None
        
    X = np.column_stack(features)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    print(f"Training GaussianHMM/GMM with {n_components} states...")
    try:
        from hmmlearn.hmm import GaussianHMM
        model_class = GaussianHMM
        model_kwargs = {"n_components": n_components, "covariance_type": "diag", "n_iter": 100}
    except ImportError:
        print("[WARNING] hmmlearn not found (Python 3.14 wheel issue). Falling back to sklearn GaussianMixture...")
        from sklearn.mixture import GaussianMixture
        model_class = GaussianMixture
        model_kwargs = {"n_components": n_components, "covariance_type": "diag", "max_iter": 100}
        
    seeds_to_test = [42, 999, 100, 200, 500]
    best_model = None
    best_score = -np.inf
    
    print("\n[DIAGNOSTIC] Evaluating EM Initializations:")
    for seed in seeds_to_test:
        kwargs = model_kwargs.copy()
        kwargs["random_state"] = seed
        model = model_class(**kwargs)
        model.fit(X_scaled)
        score = model.score(X_scaled)
        
        # sklearn's GMM returns average log-likelihood, hmmlearn returns total. Normalize for print
        print(f"  Seed {seed:<4} -> Log-Likelihood: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            
    print(f"\n[SELECTION] Selected Seed {best_model.random_state} with highest Log-Likelihood: {best_score:.4f}")
    model = best_model
    
    print("\nHMM Training complete. Temporal state-transition matrix learned successfully.")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': model, 'mean': X_mean, 'std': X_std}, 'models/hmm_regime.pkl')
    print("Saved True HMM Regime model to models/hmm_regime.pkl")
    
    states = model.predict(X_scaled)
    unique_states, counts = np.unique(states, return_counts=True)
    
    print("\nState Distribution:")
    for s, c in zip(unique_states, counts):
        print(f"  State {s}: {c} ticks ({c/len(states):.1%})")
        
    if hasattr(model, 'transmat_'):
        print("\nState Transition Matrix (Temporal Persistence):")
        print(model.transmat_)
    else:
        print("\n[INFO] GaussianMixture used (no temporal transition matrix available).")
    
    print("\n[DIAGNOSTIC] HMM State Parameters:")
    print(f"Features Used: {['realized_vol' if 'realized_vol' in df.columns else 'mid_price_vol', 'spread_bps' if 'spread_bps' in df.columns else 'ones']}")
    print(f"Random State: {model.random_state}")
    print(f"Covariance Type: {model.covariance_type}")
    
    for i in range(model.n_components):
        print(f"\nState {i}:")
        print(f"  Means: {model.means_[i]}")
        covars = getattr(model, 'covars_', getattr(model, 'covariances_', None))
        print(f"  Covars: {covars[i] if covars is not None else 'N/A'}")
        
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/features.csv')
    parser.add_argument('--states', type=int, default=3)
    args = parser.parse_args()
    
    df = load_data(args.data, max_rows=500_000)
    train_hmm(df, n_components=args.states)

if __name__ == "__main__":
    main()
