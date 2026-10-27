"""Accuracy metrics used throughout (doc section 6).

Per-sample relative error |I_ml - I_spice| / |I_spice| in %, summarised by
the median and the 95th percentile, plus the mean (the doc's "99 % mean
accuracy" target corresponds to a mean error of about 1 %). R^2 is computed
on the transformed (log-like) target, where it is meaningful across decades.
"""
import numpy as np
from sklearn.metrics import r2_score


def rel_err(pred, true):
    return np.abs(pred - true) / np.abs(true) * 100


def summary(err):
    err = np.asarray(err).ravel()
    return {"median": float(np.median(err)), "p95": float(np.percentile(err, 95)),
            "mean": float(err.mean()), "within_1pct": float((err < 1).mean() * 100)}


def r2(y_true, y_pred):
    """R^2 per output column on the transformed target."""
    return np.array([r2_score(y_true[:, k], y_pred[:, k]) for k in range(y_true.shape[1])])
