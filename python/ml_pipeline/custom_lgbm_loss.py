import numpy as np
import lightgbm as lgb

def make_asymmetric_objective(confidence_band: float, fp_penalty: float = 2.0, fn_penalty: float = 1.0):
    """Factory: returns a LightGBM-compatible fobj(preds, train_data) -> (grad, hess)."""
    def objective(preds: np.ndarray, train_data: lgb.Dataset):
        y = train_data.get_label()
        residual = y - preds

        would_trade      = np.abs(preds) > confidence_band
        wrong_direction  = np.sign(preds) != np.sign(y)
        is_false_positive = would_trade & wrong_direction

        w = np.where(is_false_positive, fp_penalty, fn_penalty)

        # loss_i = w_i * (y_i - pred_i)^2
        grad = -2.0 * w * residual
        hess = 2.0 * w
        return grad, hess
    return objective

def make_asymmetric_eval(confidence_band: float, fp_penalty: float = 2.0, fn_penalty: float = 1.0):
    """Matching eval metric so early-stopping optimizes the SAME objective it's trained on."""
    def eval_fn(preds: np.ndarray, train_data: lgb.Dataset):
        y = train_data.get_label()
        residual = y - preds
        would_trade       = np.abs(preds) > confidence_band
        wrong_direction   = np.sign(preds) != np.sign(y)
        is_false_positive = would_trade & wrong_direction
        w = np.where(is_false_positive, fp_penalty, fn_penalty)
        loss = float(np.mean(w * residual ** 2))
        return 'asym_mse', loss, False  # False = lower is better
    return eval_fn
