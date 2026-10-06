"""Hybrid retrieval experiments on CACHED embeddings (no re-encoding).

Compares pure-semantic vs hybrid (semantic + structured clinical similarity)
and k values by val AUC(rag_rate). Writes the winning RAG columns into
outputs/split_*.parquet. Run: python -m scripts.experiment_retrieval
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_raw, clean
from src.notes import add_notes
from src.split import stratified_split
from src.rag_retrieval import RagRetriever

OUT = ROOT / 'outputs'
EMB_CACHE = {s: OUT / f'emb_{s}.npy' for s in ('train', 'val', 'test')}
CHUNK = 2048

AGE_BUCKETS = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
               '[50-60)', '[60-70)', '[70-80)', '[80-100)']
A1C_MAP = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
GLU_MAP = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}


def structured_matrix(df: pd.DataFrame) -> np.ndarray:
    age = pd.Categorical(df['age'], categories=AGE_BUCKETS, ordered=True).codes.astype(float)
    a1c = df['A1Cresult'].map(A1C_MAP).fillna(0).astype(float)
    glu = df['max_glu_serum'].map(GLU_MAP).fillna(0).astype(float)
    change = (df['change'] == 'Ch').astype(float)
    dmed = (df['diabetesMed'] == 'Yes').astype(float)
    ins = df['insulin'].astype(str).str.lower().isin(('steady', 'up', 'down')).astype(float)
    num = df[['number_inpatient', 'number_emergency', 'number_outpatient',
              'time_in_hospital', 'num_medications', 'num_lab_procedures',
              'num_procedures', 'number_diagnoses']].astype(float)
    M = pd.DataFrame({'age_ord': age, 'a1c': a1c, 'glu': glu, 'change': change,
                      'dmed': dmed, 'insulin': ins})
    return pd.concat([num.reset_index(drop=True), M.reset_index(drop=True)], axis=1).to_numpy()


@torch.no_grad()
def topk_feats(Q: torch.Tensor, T: torch.Tensor, ytr: np.ndarray, k: int, loo: bool):
    sims_l, idx_l = [], []
    for s in range(0, len(Q), CHUNK):
        sim = Q[s:s + CHUNK] @ T.t()
        sims, idx = torch.topk(sim, k + 1 if loo else k, dim=1)
        sims_l.append(sims.cpu())
        idx_l.append(idx.cpu())
        del sim, sims, idx
    sims = torch.cat(sims_l).numpy()
    idx = torch.cat(idx_l).numpy()
    if loo:
        sims, idx = sims[:, 1:], idx[:, 1:]
    return np.column_stack([ytr[idx].mean(1), sims.mean(1), sims.max(1)])


def norm(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)


def main():
    print('loading splits + building current notes…', flush=True)
    df = add_notes(clean(load_raw()))
    train, val, test = stratified_split(df)
    splits = {'train': train, 'val': val, 'test': test}

    # encode + cache (only missing)
    missing = [s for s in splits if not EMB_CACHE[s].exists()]
    if missing:
        rag = RagRetriever(k=5)
        rag.fit(train['note'].tolist(), train['label'].to_numpy())
        assert rag.mode == 'sbert', 'need sbert for embedding cache'
        embs = {'train': rag._train_emb}
        for s in ('val', 'test'):
            raw = rag._embed_sbert(splits[s]['note'].tolist())
            embs[s] = norm(raw)
            del raw
        for s in splits:
            np.save(EMB_CACHE[s], embs[s].numpy())
            print(f'cached {s}', flush=True)
        del rag, embs
    E = {s: torch.from_numpy(np.load(EMB_CACHE[s])).float() for s in splits}
    ytr = train['label'].to_numpy()
    yva = val['label'].to_numpy()

    S = {s: structured_matrix(splits[s]) for s in splits}
    scaler = StandardScaler().fit(S['train'])
    Sz = {s: torch.from_numpy(scaler.transform(S[s]).astype(np.float32)) for s in splits}

    cands = {}
    cands['sem_k5'] = (E['val'], E['train'])
    # hybrid blends
    for w in (0.2, 0.35, 0.5):
        Q = norm(torch.cat([E['val'] * w, Sz['val'] * (1 - w)], dim=1))
        T = norm(torch.cat([E['train'] * w, Sz['train'] * (1 - w)], dim=1))
        cands[f'hyb{w}_k10'] = (Q, T)
    results = {}
    for name, (Q, T) in cands.items():
        k = 10 if 'k10' in name else 5
        f = topk_feats(Q, T, ytr, k, loo=False)
        auc = roc_auc_score(yva, f[:, 0])
        results[name] = (auc, f)
        print(f'{name}: val AUC(rag_rate)={auc:.4f} sim_mean={f[:,1].mean():.4f}', flush=True)
    # also try k variants for the best family
    best = max(results, key=lambda n: results[n][0])
    print('best:', best, flush=True)
    fam = 'hyb' if best.startswith('hyb') else 'sem'
    w = float(best.split('hyb')[1].split('_')[0]) if fam == 'hyb' else None
    for k in (5, 20):
        if fam == 'hyb':
            Q = norm(torch.cat([E['val'] * w, Sz['val'] * (1 - w)], dim=1))
            T = norm(torch.cat([E['train'] * w, Sz['train'] * (1 - w)], dim=1))
        else:
            Q, T = E['val'], E['train']
        f = topk_feats(Q, T, ytr, k, loo=False)
        auc = roc_auc_score(yva, f[:, 0])
        results[f'{fam}_k{k}'] = (auc, f)
        print(f'{fam}_k{k}: val AUC(rag_rate)={auc:.4f}', flush=True)
    best = max(results, key=lambda n: results[n][0])
    print('WINNER:', best, results[best][0], flush=True)

    # materialize winner for all splits (train with LOO)
    if fam == 'hyb':
        Qtr = norm(torch.cat([E['train'] * w, Sz['train'] * (1 - w)], dim=1))
        Qva = norm(torch.cat([E['val'] * w, Sz['val'] * (1 - w)], dim=1))
        Qte = norm(torch.cat([E['test'] * w, Sz['test'] * (1 - w)], dim=1))
        T = norm(torch.cat([E['train'] * w, Sz['train'] * (1 - w)], dim=1))
    else:
        Qtr, Qva, Qte, T = E['train'], E['val'], E['test'], E['train']
    k = int(best.split('_k')[1])
    feats = {'train': topk_feats(Qtr, T, ytr, k, loo=True),
             'val': topk_feats(Qva, T, ytr, k, loo=False),
             'test': topk_feats(Qte, T, ytr, k, loo=False)}
    for s in splits:
        for j, c in enumerate(RagRetriever.columns()):
            splits[s][c] = feats[s][:, j]
        splits[s].to_parquet(OUT / f'split_{s}.parquet', index=False)
    (OUT / 'rag_config.txt').write_text(f'{best} w={w} k={k}\n')
    print('splits overwritten with', best, flush=True)


if __name__ == '__main__':
    main()
