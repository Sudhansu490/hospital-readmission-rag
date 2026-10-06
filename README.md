# Hospital Readmission Risk Prediction (Supervised + RAG Feature Engineering)

Predict **30-day readmission at discharge** on the Diabetes 130-US Hospitals
1999–2008 dataset (~101,766 encounters, 11.2% positive), with a genuine
RAG-style retrieval step: for each patient, retrieve similar past encounters
and use their historical readmission rate as an engineered feature. Compare
Logistic Regression, Random Forest, and a fine-tuned Transformer, each
with/without the RAG feature, on AUC-ROC, F1, Precision, Recall.

## Results (test set, thresholds tuned on val for max F1)

| Model | RAG | AUC | F1 | Precision | Recall | Thr |
|---|---|---|---|---|---|---|
| LogReg | no | 0.6440 | 0.2559 | 0.1675 | 0.5423 | 0.50 |
| LogReg | **yes** | 0.6448 | 0.2549 | 0.1667 | 0.5417 | 0.50 |
| Random Forest | no | **0.6767** | 0.2777 | 0.1847 | 0.5587 | 0.40 |
| Random Forest | **yes** | 0.6745 | 0.2805 | 0.1939 | 0.5070 | 0.40 |
| Tiny-BERT fine-tuned | no | 0.5975 | 0.2230 | 0.1297 | 0.7940 | 0.10 |
| Tiny-BERT fine-tuned | **yes** | 0.5959 | 0.2008 | 0.1116 | 1.0000 | 0.05 |

(Val metrics and thresholds in `outputs/metrics.json`.)

**Reading:** the RAG feature is *honestly neutral* here (±0.002 AUC).
Standalone, `rag_rate` reaches AUC ≈ 0.61 — but it is built from the same
clinical factors the tabular models already see, so it adds no orthogonal
signal for LogReg/RF. For the text-only tiny-BERT, the appended rate sentence
likewise can't compensate for a 4.4M-param model's capacity limits on this
task. The retrieval machinery, ablations, and leakage guards below are the
reusable contribution; see Limitations.

## Method

1. **Label:** `readmitted == "<30"` → 1, else 0. Drop ID/near-empty cols
   (`encounter_id`, `patient_nbr`, `weight`, `payer_code`, `medical_specialty`);
   `?` → `Missing`. Stratified 70/15/15 split (`src/split.py`).
2. **Synthetic notes (`src/notes.py`):** deterministic, value-conditioned
   discharge summaries (prior utilization, stay length, diagnoses, A1C/glucose
   phrasing, meds incl. insulin, regimen change). No label use, no randomness.
3. **RAG retrieval:** hybrid similarity over train-only index —
   `0.5 × cosine(MiniLM note embeddings) + 0.5 × standardized structured
   clinical vector` (prior inpatient/ER/outpatient, LOS, meds, labs,
   procedures, diagnosis count, A1C/glucose ordinals, change, diabetesMed,
   insulin, age bucket), top-k by torch chunked cosine. Features per query:
   `rag_rate` (neighbor readmission mean), `rag_mean_sim`, `rag_max_sim`.
   - Leakage guards: index fit on train only; train rows use leave-one-out
     (self excluded); val/test query the frozen index (`tests/test_rag_feature.py`).
   - Selection: grid over {pure-semantic, hybrid w, k} by **val AUC(rag_rate)**:
     pure k=5 → 0.545; hybrid w=0.5 k=100 → **0.613** (`experiment_retrieval.py`,
     `sweep2.py`, `outputs/rag_config.txt`). Pure TF-IDF+SVD fallback if the
     embedding model can't load.
4. **Models:** LogReg (`balanced`, C default) and RF (300 trees,
   `balanced_subsample`) on one-hot + scaled tabular features ± 3 RAG cols;
   tiny-BERT (`google/bert_uncased_L-2_H-128_A-2`, 8k/2k stratified subsample,
   2 epochs) on note text, +RAG arm appends
   `"Similar past cases readmission rate: X.XX."` to the note.
   Operating thresholds tuned on val (max F1), applied to test.
5. **Why tiny-BERT, not DistilBERT:** DistilBERT benched at ~134 min/arm on
   this 2-thread CPU box; tiny-BERT trains in ~4 min/arm with the same
   Trainer protocol. Documented trade-off, not a silent swap.

## Layout

```
pyproject.toml         # deps + pytest config (pip install -e . / .[dev])
data/diabetic_data.csv # working copy (UCI Diabetes 130-US Hospitals)
src/                   # config, data_loader, notes, split, rag_retrieval,
                       # features, train_baselines, train_transformer,
                       # metrics, run_all
scripts/               # smoke (3k-row e2e), experiment_retrieval + sweep2
                       # (RAG method selection on cached embeddings)
outputs/               # metrics.json, rag_config.txt, split_*.parquet (cache),
                       # emb_*.npy (cache), models/ (saved estimators, gitignored)
tests/                 # leakage/shape guards, pipeline unit tests
```

## Reproduce

```powershell
pip install -e .[dev]
python -m pytest tests -q          # 7 tests, ~15 s
python -m scripts.smoke            # fast pipeline smoke test
python -m src.run_all --rag-only --rebuild   # rebuild retrieval index (~35 min CPU)
python -m src.run_all --skip-transformer     # baselines ±RAG from cached splits
python -m src.run_all --transformer-only --bert-norag   # one BERT arm (~5 min)
python -m src.run_all --transformer-only --bert-rag     # other BERT arm
python -m src.run_all              # everything (uses cache where present)
```

## Limitations & next steps

- Single-site-era tabular data with synthetic (not real) notes; template
  wording bounds semantic diversity — real discharge text would likely show a
  larger RAG effect.
- RAG signal overlaps tabular features by construction; future work: retrieve
  on genuinely held-out modalities, or use neighbor *texts* as few-shot LLM context.
- Tiny-BERT underfits (recall-heavy operating points); with a GPU, rerun the
  same protocol with DistilBERT/ClinicalBERT at full data.
- No probability calibration; no fairness audit across race/gender/age.
