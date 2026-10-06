"""Shared metrics: AUC-ROC, F1, Precision, Recall + val threshold tuning."""
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score)


def binary_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def tune_threshold(y_true, y_score) -> float:
    """Pick the threshold maximizing F1 on validation scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_true, (y_score >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)
