"""Tabular feature matrix: one-hot categoricals + scaled numerics, + optional RAG cols."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NOTE_COL = "note"
LABEL_COL = "label"
RAG_COLS = ["rag_rate", "rag_mean_sim", "rag_max_sim"]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    feat_cols = [c for c in df.columns if c not in (NOTE_COL, LABEL_COL, *RAG_COLS)]
    # non-numeric -> categorical (robust to pandas str/object dtypes)
    cat = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in feat_cols if c not in cat]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    pre.feature_names_ = feat_cols  # type: ignore
    return pre


def to_matrix(pre: ColumnTransformer, df: pd.DataFrame, use_rag: bool):
    X = pre.transform(df[pre.feature_names_])  # type: ignore
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    if use_rag:
        rag = df[RAG_COLS].to_numpy(dtype=float)
        rag = np.nan_to_num(rag, nan=0.0)
        X = np.hstack([X, rag])
    return X
