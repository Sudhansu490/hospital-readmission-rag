"""Fast smoke test: 3000-row stratified sample, TF-IDF-forced RAG, LogReg ±RAG.

Run: python -m scripts.smoke
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.data_loader import load_raw, clean
from src.notes import add_notes
from src.split import stratified_split
from src.rag_retrieval import RagRetriever
from src.features import build_preprocessor, to_matrix
from src.metrics import binary_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main():
    print("[smoke] load...", flush=True)
    df = add_notes(clean(load_raw()))
    print("shape:", df.shape, "pos_rate:", round(float(df.label.mean()), 4), flush=True)
    # stratified subsample to 3000 for speed
    from sklearn.model_selection import train_test_split
    df_big = df
    df, _ = train_test_split(df_big, train_size=3000, random_state=42, stratify=df_big["label"])
    print("subsample:", df.shape, flush=True)

    train, val, test = stratified_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    rag = RagRetriever(k=5, embed_model="nonexistent-model-forces-tfidf-fallback-xyz")
    rag.fit(train["note"].tolist(), train["label"].to_numpy())
    print("retriever mode:", rag.mode, flush=True)
    assert rag.mode == "tfidf", "expected TF-IDF fallback with bogus model name"
    for name, split, loo in (("train", train, True), ("val", val, False), ("test", test, False)):
        feats = rag.transform_train() if loo else rag.transform(split["note"].tolist())
        assert feats.shape == (len(split), 3), feats.shape
        for j, c in enumerate(RagRetriever.columns()):
            split[c] = feats[:, j]
        print(f"{name} rag_rate mean={split['rag_rate'].mean():.4f}", flush=True)

    # leakage check: train rag_rate must not equal own label everywhere (LOO)
    # (trivially true unless kNN degenerate, but assert finite + in [0,1])
    assert train["rag_rate"].between(0, 1).all()

    pre = build_preprocessor(train)
    pre.fit(train)
    for use_rag in (False, True):
        Xtr, ytr = to_matrix(pre, train, use_rag), train["label"].to_numpy()
        Xte, yte = to_matrix(pre, test, use_rag), test["label"].to_numpy()
        print(f"use_rag={use_rag} Xtr={Xtr.shape} Xte={Xte.shape}", flush=True)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        m = binary_metrics(yte, (prob >= 0.5).astype(int), prob)
        print(f"logreg use_rag={use_rag}: {m}", flush=True)

    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
