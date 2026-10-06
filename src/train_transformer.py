"""Fine-tune DistilBERT on note text, with/without RAG sentence appended.

CPU-friendly: trains on a stratified subsample (see config TR_SUBSAMPLE_*).
Evaluates on val + full test notes for the reported metrics.
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

from .config import (RANDOM_STATE, MODELS, TR_SUBSAMPLE_TRAIN, TR_SUBSAMPLE_VAL,
                     TR_EPOCHS, TR_MODEL, TR_MAX_LEN, TR_BATCH)
from .notes import with_rag_sentence
from .metrics import binary_metrics, tune_threshold


def _prep(df, use_rag: bool):
    texts = (df.apply(lambda r: with_rag_sentence(r["note"], r["rag_rate"]), axis=1)
             if use_rag else df["note"]).tolist()
    return texts, df["label"].to_numpy()


def _subsample(texts, labels, n, seed=RANDOM_STATE):
    if len(texts) <= n:
        return texts, labels
    idx, _ = train_test_split(range(len(texts)), train_size=n, random_state=seed,
                              stratify=labels)
    idx = sorted(idx)
    return [texts[i] for i in idx], labels[idx]


def train_transformer(train, val, test, use_rag: bool):
    from datasets import Dataset
    tag = "withRAG" if use_rag else "noRAG"
    tr_texts, tr_y = _subsample(*_prep(train, use_rag), TR_SUBSAMPLE_TRAIN)
    va_texts, va_y = _subsample(*_prep(val, use_rag), TR_SUBSAMPLE_VAL)
    te_texts, te_y = _prep(test, use_rag)

    # WordPiece tiny-BERT (fast tokenizer, no sentencepiece needed).
    tok = AutoTokenizer.from_pretrained(TR_MODEL)

    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=TR_MAX_LEN)

    ds_tr = Dataset.from_dict({"text": tr_texts, "label": tr_y}).map(tok_fn, batched=True)
    ds_va = Dataset.from_dict({"text": va_texts, "label": va_y}).map(tok_fn, batched=True)
    ds_te = Dataset.from_dict({"text": te_texts, "label": te_y}).map(tok_fn, batched=True)
    for ds in (ds_tr, ds_va, ds_te):
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(TR_MODEL, num_labels=2)
    args = TrainingArguments(
        output_dir=str(MODELS / f"bert_{tag}"), seed=RANDOM_STATE,
        num_train_epochs=TR_EPOCHS, per_device_train_batch_size=TR_BATCH,
        per_device_eval_batch_size=TR_BATCH * 2,
        eval_strategy="epoch", save_strategy="epoch",
        logging_strategy="no", report_to="none",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr,
                      eval_dataset=ds_va,
                      data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    out = {}
    probs = {}
    for split, ds in (("val", ds_va), ("test", ds_te)):
        logits = trainer.predict(ds).predictions
        probs[split] = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    # operating threshold tuned on VAL (max F1), applied to test
    thr = tune_threshold(va_y, probs["val"])
    for (split, y) in (("val", va_y), ("test", te_y)):
        pred = (probs[split] >= thr).astype(int)
        out[f"bert_{tag}_{split}"] = {**binary_metrics(y, pred, probs[split]),
                                     "threshold": thr}
        print(f"[bert] {tag} {split} thr={thr:.2f}: {out[f'bert_{tag}_{split}']}")
    trainer.save_model(str(MODELS / f"bert_{tag}_final"))
    return out
