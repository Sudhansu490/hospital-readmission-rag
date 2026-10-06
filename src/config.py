"""Shared config for hospital readmission RAG project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "diabetic_data.csv"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "outputs" / "models"
MODELS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of remaining after test split -> ~70/15/15 overall

LABEL_POS = "<30"  # 30-day readmission positive class

# Columns dropped: near-empty or IDs (standard for this dataset)
DROP_COLS = ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"]

EMBED_MODEL = "all-MiniLM-L6-v2"
# Hybrid retrieval: 0.5 * note-embedding cosine + 0.5 * structured clinical
# similarity (standardized), k=100 neighbors -> rag_rate (+ sim stats).
# Chosen by val-AUC sweep over {pure-semantic, hybrid w, k} (see
# scripts/experiment_retrieval.py, scripts/sweep2.py, outputs/rag_config.txt).
RAG_K = 100
RAG_W_SEM = 0.5

# Transformer budget: tiny BERT (L2/H128, ~4.4M params) fine-tune,
# CPU-feasible on 2 threads (~10x faster than DistilBERT, which benched at
# ~134 min/arm on this box). Still a genuine fine-tuned-Transformer +/-RAG
# comparison; model id recorded in outputs for reproducibility.
TR_SUBSAMPLE_TRAIN = 8000
TR_SUBSAMPLE_VAL = 2000
TR_EPOCHS = 2
TR_MODEL = "google/bert_uncased_L-2_H-128_A-2"
TR_MAX_LEN = 256
TR_BATCH = 32
