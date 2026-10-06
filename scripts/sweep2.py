"""Second-stage RAG sweep over hybrid weight w and k (uses cached embeddings).

Run: python -m scripts.sweep2
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiment_retrieval import structured_matrix, norm, topk_feats, EMB_CACHE

import warnings
warnings.filterwarnings('ignore')


def main():
    tr = pd.read_parquet(ROOT / 'outputs/split_train.parquet')
    va = pd.read_parquet(ROOT / 'outputs/split_val.parquet')
    te = pd.read_parquet(ROOT / 'outputs/split_test.parquet')
    E = {s: torch.from_numpy(np.load(EMB_CACHE[s])).float() for s in ('train', 'val', 'test')}
    S = {s: structured_matrix(df) for s, df in (('train', tr), ('val', va), ('test', te))}
    scaler = StandardScaler().fit(S['train'])
    Sz = {s: torch.from_numpy(scaler.transform(S[s]).astype(np.float32)) for s in S}
    ytr, yva, yte = tr['label'].to_numpy(), va['label'].to_numpy(), te['label'].to_numpy()

    best = (-1, None)
    for w in (0.35, 0.5, 0.65):
        Q = norm(torch.cat([E['val'] * w, Sz['val'] * (1 - w)], dim=1))
        T = norm(torch.cat([E['train'] * w, Sz['train'] * (1 - w)], dim=1))
        for k in (20, 30, 50, 100):
            f = topk_feats(Q, T, ytr, k, loo=False)
            auc = roc_auc_score(yva, f[:, 0])
            print(f'w={w} k={k}: val AUC={auc:.4f}', flush=True)
            if auc > best[0]:
                best = (auc, (w, k, f))
    auc, (w, k, _) = best
    print('WINNER w=%s k=%d auc=%.4f' % (w, k, auc), flush=True)
    T = norm(torch.cat([E['train'] * w, Sz['train'] * (1 - w)], dim=1))
    Qtr = norm(torch.cat([E['train'] * w, Sz['train'] * (1 - w)], dim=1))
    Qva = norm(torch.cat([E['val'] * w, Sz['val'] * (1 - w)], dim=1))
    Qte = norm(torch.cat([E['test'] * w, Sz['test'] * (1 - w)], dim=1))
    F = {'train': topk_feats(Qtr, T, ytr, k, loo=True),
         'val': topk_feats(Qva, T, ytr, k, loo=False),
         'test': topk_feats(Qte, T, ytr, k, loo=False)}
    from src.rag_retrieval import RagRetriever
    for s, df in (('train', tr), ('val', va), ('test', te)):
        for j, c in enumerate(RagRetriever.columns()):
            df[c] = F[s][:, j]
        df.to_parquet(ROOT / f'outputs/split_{s}.parquet', index=False)
        auc_t = roc_auc_score(df['label'].to_numpy(), F[s][:, 0])
        print(s, 'wrote, AUC(rag)=%.4f' % auc_t, flush=True)
    (ROOT / 'outputs/rag_config.txt').write_text(f'hybrid w={w} k={k} val_auc={auc:.4f}\n')
    print('SWEEP2 DONE', flush=True)


if __name__ == '__main__':
    main()
