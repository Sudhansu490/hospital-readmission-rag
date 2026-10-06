"""Logistic Regression + Random Forest, each with/without RAG features."""
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .config import RANDOM_STATE, MODELS
from .features import build_preprocessor, to_matrix
from .metrics import binary_metrics, tune_threshold


def train_baselines(train, val, test):
    t0 = time.time()
    pre = build_preprocessor(train)
    pre.fit(train)
    print(f"[baselines] preprocessor fit done ({time.time()-t0:.1f}s)", flush=True)
    out = {}
    specs = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=RANDOM_STATE),
    }
    for use_rag in (False, True):
        suffix = "withRAG" if use_rag else "noRAG"
        t1 = time.time()
        Xtr, ytr = to_matrix(pre, train, use_rag), train["label"].to_numpy()
        Xva, yva = to_matrix(pre, val, use_rag), val["label"].to_numpy()
        Xte, yte = to_matrix(pre, test, use_rag), test["label"].to_numpy()
        print(f"[baselines] matrices {suffix}: train={Xtr.shape} "
              f"({time.time()-t1:.1f}s)", flush=True)
        for name, clf in specs.items():
            t2 = time.time()
            clf.fit(Xtr, ytr)
            key = f"{name}_{suffix}"
            # operating threshold tuned on VAL (max F1), applied to test
            thr = tune_threshold(yva, clf.predict_proba(Xva)[:, 1])
            for split, X, y in (("val", Xva, yva), ("test", Xte, yte)):
                prob = clf.predict_proba(X)[:, 1]
                pred = (prob >= thr).astype(int)
                out[f"{key}_{split}"] = {**binary_metrics(y, pred, prob),
                                        "threshold": thr}
            joblib.dump(clf, MODELS / f"{key}.joblib")
            print(f"[baselines] {key}: thr={thr:.2f} test={out[f'{key}_test']} "
                  f"({time.time()-t2:.1f}s)", flush=True)
    joblib.dump(pre, MODELS / "preprocessor.joblib")
    return out
