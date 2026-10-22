import numpy as np


# ─────────────────────────────────────────────────────────────────
# Core posterior models
# ─────────────────────────────────────────────────────────────────
class BayesianLinearArm:
    """Online Bayesian ridge regression posterior: N(mean, cov) over theta,
    reward ~ theta . x + N(0, noise_var)."""
    def __init__(self, n_features, prior_var=4.0, noise_var=1.0):
        self.noise_var = noise_var
        self.precision = (1.0 / prior_var) * np.eye(n_features)
        self._b = np.zeros(n_features)  # accumulator: sum(x*y)/noise_var (+ prior term, 0 here)
        self.mean = np.zeros(n_features)
        self.cov = np.linalg.inv(self.precision)

    def update(self, x, y):
        x = np.asarray(x, dtype=float)
        self.precision += np.outer(x, x) / self.noise_var
        self._b += x * y / self.noise_var
        self.cov = np.linalg.inv(self.precision)
        self.mean = self.cov @ self._b

    def sample_theta(self, rng):
        return rng.multivariate_normal(self.mean, self.cov)

    def predict_mean(self, x):
        return float(np.dot(self.mean, x))

    def ucb(self, x, beta=2.0):
        """Drop-in LinUCB alternative to sample_theta + dot product: same
        posterior, upper-confidence-bound selection instead of sampling."""
        x = np.asarray(x, dtype=float)
        mean = float(np.dot(self.mean, x))
        conf = beta * np.sqrt(float(x @ self.cov @ x))
        return mean + conf


class BetaBernoulliFill:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def update(self, filled: bool):
        if filled:
            self.alpha += 1
        else:
            self.beta += 1

    def sample(self, rng):
        return rng.beta(self.alpha, self.beta)

    def mean(self):
        return self.alpha / (self.alpha + self.beta)


# ─────────────────────────────────────────────────────────────────
# Bandit orchestrator
# ─────────────────────────────────────────────────────────────────
class SpreadBandit:
    def __init__(self, arms, n_context_features, n_regime_buckets,
                 baseline_arm_idx=0, safe_arm_idx=None, max_exploration_frac=0.2,
                 conservative_alpha=0.3, gap_ema_decay=0.995,
                 gap_trigger_threshold=0.05, cooldown_steps=200,
                 safe_inventory_threshold=0.6, noise_var=1.0, prior_var=4.0,
                 rng_seed=0):
        self.arms = arms
        self.n_arms = len(arms)
        self.baseline_arm_idx = baseline_arm_idx
        self.safe_arm_idx = safe_arm_idx if safe_arm_idx is not None else int(np.argmax(arms))
        self.max_exploration_frac = max_exploration_frac
        self.conservative_alpha = conservative_alpha
        self.gap_ema_decay = gap_ema_decay
        self.gap_trigger_threshold = gap_trigger_threshold
        self.cooldown_steps = cooldown_steps
        self.safe_inventory_threshold = safe_inventory_threshold
        self.rng = np.random.default_rng(rng_seed)

        self.markout_models = [BayesianLinearArm(n_context_features, prior_var, noise_var)
                                for _ in range(self.n_arms)]
        self.fill_models = [[BetaBernoulliFill() for _ in range(n_regime_buckets)]
                             for _ in range(self.n_arms)]

        self.reward_gap_ema = 0.0
        self.forced_baseline_cooldown = 0
        self.n_forced_cooldowns = 0

    def select_arm(self, context, regime_bucket, inventory_frac, kill_switch_active):
        # ── Hard gates: fall back to the SAFE arm (widest), not baseline ──
        if kill_switch_active or abs(inventory_frac) > self.safe_inventory_threshold:
            return self.safe_arm_idx, False
        if self.forced_baseline_cooldown > 0:
            self.forced_baseline_cooldown -= 1
            return self.safe_arm_idx, False
        # ── Hard exploration-fraction ceiling ──
        if self.rng.random() > self.max_exploration_frac:
            return self.baseline_arm_idx, False

        # ── Thompson Sampling over the exploration slice ──
        values = np.empty(self.n_arms)
        for a in range(self.n_arms):
            p_fill = self.fill_models[a][regime_bucket].sample(self.rng)
            theta = self.markout_models[a].sample_theta(self.rng)
            values[a] = p_fill * float(np.dot(theta, context))
        chosen = int(np.argmax(values))
        return chosen, (chosen != self.baseline_arm_idx)

    def update(self, arm_idx, context, regime_bucket, filled: bool, markout: float = 0.0):
        self.fill_models[arm_idx][regime_bucket].update(filled)
        realized_reward = 0.0
        if filled:
            self.markout_models[arm_idx].update(context, markout)
            realized_reward = markout

        p_fill_base = self.fill_models[self.baseline_arm_idx][regime_bucket].mean()
        markout_base_mean = self.markout_models[self.baseline_arm_idx].predict_mean(context)
        baseline_est_reward = p_fill_base * markout_base_mean

        gap_t = realized_reward - (1 - self.conservative_alpha) * baseline_est_reward
        self.reward_gap_ema = (self.gap_ema_decay * self.reward_gap_ema
                                + (1 - self.gap_ema_decay) * gap_t)

        if self.reward_gap_ema < -self.gap_trigger_threshold:
            self.forced_baseline_cooldown = self.cooldown_steps
            self.reward_gap_ema = 0.0
            self.n_forced_cooldowns += 1
