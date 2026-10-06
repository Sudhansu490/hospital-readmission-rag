"""Orchestrator: load -> notes -> split -> RAG -> baselines -> transformer -> report."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ^ allows `python src/run_all.py` as well as `python -m src.run_all`

from src.config import OUTPUTS, RAG_K, EMBED_MODEL  # noqa: E402
from src.data_loader import load_raw, clean  # noqa: E402
from src.notes import add_notes  # noqa: E402
from src.split import stratified_split  # noqa: E402
from src.rag_retrieval import RagRetriever  # noqa: E402
from src.train_baselines import train_baselines  # noqa: E402
from src.train_transformer import train_transformer  # noqa: E402


def load_or_build_splits(force_rebuild=False):
    """Load cached splits with RAG features, or build + cache them."""
    import pandas as pd
    cache = {s: OUTPUTS / f"split_{s}.parquet" for s in ("train", "val", "test")}
    if not force_rebuild and all(p.exists() for p in cache.values()):
        print("loading cached splits with RAG features…", flush=True)
        return (pd.read_parquet(cache["train"]), pd.read_parquet(cache["val"]),
                pd.read_parquet(cache["test"]))
    print("[1/5] loading + cleaning…", flush=True)
    df = add_notes(clean(load_raw()))
    print("  shape:", df.shape, "pos_rate:", round(df.label.mean(), 4), flush=True)

    print("[2/5] splitting (train-only retrieval index)…", flush=True)
    train, val, test = stratified_split(df)
    print(f"  train={len(train)} val={len(val)} test={len(test)}", flush=True)

    print(f"[3/5] RAG retrieval (k={RAG_K}, {EMBED_MODEL})…", flush=True)
    rag = RagRetriever(k=RAG_K, embed_model=EMBED_MODEL)
    rag.fit(train["note"].tolist(), train["label"].to_numpy())
    print("  retriever mode:", rag.mode, flush=True)
    for name, split, loo in (("train", train, True), ("val", val, False), ("test", test, False)):
        t0 = time.time()
        feats = rag.transform_train() if loo else rag.transform(split["note"].tolist())
        for j, c in enumerate(RagRetriever.columns()):
            split[c] = feats[:, j]
        print(f"  {name} rag_rate mean={split['rag_rate'].mean():.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        split.to_parquet(cache[name], index=False)
    print("  splits cached to outputs/split_*.parquet", flush=True)
    return train, val, test


def main(skip_transformer=False, baselines_only=False, transformer_only=False,
         rebuild=False, rag_only=False, bert_norag=False, bert_rag=False):
    train, val, test = load_or_build_splits(force_rebuild=rebuild)

    if rag_only:
        print("RAG splits built + cached; exiting (--rag-only)", flush=True)
        return

    if transformer_only:
        print("[5/5] transformer (fine-tuned Transformer x noRAG/withRAG)…", flush=True)
        results = load_results()
        arms = []
        if bert_norag or (not bert_norag and not bert_rag):
            arms.append(False)
        if bert_rag or (not bert_norag and not bert_rag):
            arms.append(True)
        for use_rag in arms:
            results.update(train_transformer(train, val, test, use_rag=use_rag))
        save_and_report(results)
        return

    print("[4/5] baselines (LogReg/RF x noRAG/withRAG)…", flush=True)
    results = train_baselines(train, val, test)

    if not skip_transformer and not baselines_only:
        print("[5/5] transformer (DistilBERT x noRAG/withRAG)…", flush=True)
        for use_rag in (False, True):
            results.update(train_transformer(train, val, test, use_rag=use_rag))
    else:
        print("[5/5] transformer skipped", flush=True)

    save_and_report(results)


def load_results():
    if (OUTPUTS / "metrics.json").exists():
        return json.loads((OUTPUTS / "metrics.json").read_text())
    return {}


def save_and_report(results):
    with open(OUTPUTS / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n==== ABLATION (test) ====", flush=True)
    for k, v in results.items():
        if k.endswith("_test"):
            print(f"{k:22s} AUC={v['auc']:.4f} F1={v['f1']:.4f} "
                  f"P={v['precision']:.4f} R={v['recall']:.4f}", flush=True)
    print("saved:", OUTPUTS / "metrics.json", flush=True)


if __name__ == "__main__":
    main(skip_transformer="--skip-transformer" in sys.argv,
         baselines_only="--baselines-only" in sys.argv,
         transformer_only="--transformer-only" in sys.argv,
         rebuild="--rebuild" in sys.argv,
         rag_only="--rag-only" in sys.argv,
         bert_norag="--bert-norag" in sys.argv,
         bert_rag="--bert-rag" in sys.argv)
