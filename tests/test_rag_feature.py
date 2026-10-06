"""Leakage + shape guards for the RAG retrieval step (fast TF-IDF path)."""
import numpy as np

from src.rag_retrieval import RagRetriever


def _retriever():
    # bogus model name forces the deterministic TF-IDF fallback (no download)
    return RagRetriever(k=1, embed_model="nonexistent-fallback-test-model-xyz")


def test_leave_one_out_excludes_self():
    notes = ["apple apple apple", "apple apple apple", "zebra zebra zebra"]
    labels = np.array([1, 0, 0])
    rag = _retriever().fit(notes, labels)
    assert rag.mode == "tfidf"
    feats = rag.transform_train()
    assert feats.shape == (3, 3)
    # row 0's only neighbor must be row 1 (identical note), NOT itself:
    # rag_rate == 0.0 proves own label (1) was excluded
    assert feats[0, 0] == 0.0


def test_feature_shapes_and_ranges():
    notes = [f"patient note number {i} diabetes insulin" for i in range(20)]
    labels = np.array([i % 2 for i in range(20)])
    rag = RagRetriever(k=5, embed_model="nonexistent-fallback-test-model-xyz")
    rag.fit(notes, labels)
    tr = rag.transform_train()
    q = rag.transform(notes[:7])
    assert tr.shape == (20, 3)
    assert q.shape == (7, 3)
    assert ((tr[:, 0] >= 0) & (tr[:, 0] <= 1)).all()
    assert ((q[:, 0] >= 0) & (q[:, 0] <= 1)).all()
    assert RagRetriever.columns() == ["rag_rate", "rag_mean_sim", "rag_max_sim"]
    # index was fit on train only
    assert rag._train_emb.shape[0] == 20
