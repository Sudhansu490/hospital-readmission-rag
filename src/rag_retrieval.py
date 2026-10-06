"""RAG-style retrieval: embed notes, top-k similar train cases -> readmission rate.

Leakage guards:
  * index (embeddings / neighbors) fit on TRAIN notes only
  * train rows use leave-one-out (top k+1, drop self)
  * val/test query the frozen train index
Features per query: rag_rate (mean label of neighbors), mean_sim, max_sim.

Backend: torch chunked cosine top-k (no sklearn/OpenMP in the hot loop --
avoids thread-pool deadlocks on Windows and is faster than brute NN).
Falls back to TF-IDF (+SVD to 256-d) if sentence-transformers can't load.
"""
import numpy as np
import torch

QUERY_CHUNK = 2048


class RagRetriever:
    def __init__(self, k=5, embed_model="all-MiniLM-L6-v2"):
        self.k = k
        self.embed_model_name = embed_model
        self.mode = "tfidf"  # or 'sbert'
        self._st = None
        self._vec = None
        self._svd = None
        self._train_emb = None  # torch tensor, L2-normalized, (n_train, d)
        self._train_labels = None

    # -- embedding -----------------------------------------------------
    def _embed_sbert(self, texts):
        emb = np.asarray(self._st.encode(texts, show_progress_bar=False,
                                         convert_to_numpy=True,
                                         batch_size=256))
        return torch.from_numpy(emb).float()

    def _embed_tfidf(self, texts):
        X = self._vec.transform(texts)
        X = self._svd.transform(X)
        return torch.from_numpy(np.asarray(X, dtype=np.float32))

    def _normalize(self, t):
        return t / (t.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))

    def fit(self, train_notes: list, train_labels: np.ndarray):
        self._train_labels = torch.from_numpy(
            np.asarray(train_labels, dtype=np.float32))
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.embed_model_name)
            self.mode = "sbert"
            raw = self._embed_sbert(train_notes)
        except Exception as e:
            print(f"[rag] sbert load failed ({str(e)[:200]}); TF-IDF+SVD fallback")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self.mode = "tfidf"
            self._vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2),
                                        sublinear_tf=True)
            Xtr = self._vec.fit_transform(train_notes)
            n_comp = max(1, min(256, Xtr.shape[0] - 1, Xtr.shape[1] - 1))
            self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self._svd.fit(Xtr)
            raw = self._embed_tfidf(train_notes)
        self._train_emb = self._normalize(raw)
        del raw
        return self

    # -- top-k ----------------------------------------------------------
    @torch.no_grad()
    def _topk(self, queries: torch.Tensor, k: int):
        """Return (sims, idx) of top-k train neighbors per query, chunked."""
        queries = self._normalize(queries)
        sims_all, idx_all = [], []
        for s in range(0, len(queries), QUERY_CHUNK):
            sim = queries[s:s + QUERY_CHUNK] @ self._train_emb.t()
            sims, idx = torch.topk(sim, k, dim=1)
            sims_all.append(sims.cpu())
            idx_all.append(idx.cpu())
            del sim, sims, idx
        return torch.cat(sims_all).numpy(), torch.cat(idx_all).numpy()

    def _to_features(self, sims, idx):
        neigh = self._train_labels[idx].numpy() if isinstance(
            idx, torch.Tensor) else self._train_labels.numpy()[idx]
        rag_rate = neigh.mean(axis=1)
        return np.column_stack([rag_rate, sims.mean(axis=1), sims.max(axis=1)])

    def transform_train(self):
        """Leave-one-out features for train rows (self excluded)."""
        sims, idx = self._topk(self._train_emb, self.k + 1)
        return self._to_features(sims[:, 1:], idx[:, 1:])

    def transform(self, notes: list):
        raw = self._embed_sbert(notes) if self.mode == "sbert" \
            else self._embed_tfidf(notes)
        sims, idx = self._topk(raw, self.k)
        del raw
        return self._to_features(sims, idx)

    @staticmethod
    def columns():
        return ["rag_rate", "rag_mean_sim", "rag_max_sim"]
