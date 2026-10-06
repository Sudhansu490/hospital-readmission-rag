"""Pipeline unit tests: cleaning, notes, splits, features, metrics."""
import numpy as np
import pandas as pd

from src.data_loader import clean
from src.notes import make_note, with_rag_sentence
from src.split import stratified_split
from src.features import build_preprocessor, to_matrix
from src.metrics import binary_metrics, tune_threshold


def _raw(n=300):
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "encounter_id": range(n),
        "patient_nbr": range(n),
        "weight": ["?"] * n,
        "payer_code": ["?"] * n,
        "medical_specialty": ["?"] * n,
        "race": rng.choice(["Caucasian", "AfricanAmerican", "?"], n),
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.choice(["[50-60)", "[60-70)"], n),
        "admission_type_id": rng.randint(1, 8, n),
        "time_in_hospital": rng.randint(1, 14, n),
        "num_medications": rng.randint(1, 30, n),
        "diag_1": rng.choice(["250", "414", "428"], n),
        "A1Cresult": rng.choice(["None", "Norm", ">7", ">8"], n),
        "max_glu_serum": rng.choice(["None", "Norm", ">200"], n),
        "change": rng.choice(["No", "Ch"], n),
        "diabetesMed": rng.choice(["Yes", "No"], n),
        "insulin": rng.choice(["No", "Steady", "Up"], n),
        "number_inpatient": rng.randint(0, 3, n),
        "number_emergency": rng.randint(0, 2, n),
        "number_outpatient": rng.randint(0, 2, n),
        "readmitted": rng.choice(["NO", ">30", "<30"], n, p=[0.5, 0.39, 0.11]),
    })


def test_clean_label_and_missing():
    df = clean(_raw())
    assert "label" in df and "readmitted" not in df
    assert set(df["label"].unique()) <= {0, 1}
    assert abs(df["label"].mean() - 0.11) < 0.05
    for c in ("encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"):
        assert c not in df.columns
    assert not (df["race"] == "?").any()


def test_notes_deterministic_and_label_free():
    df = clean(_raw(50))
    row = df.iloc[0]
    n1, n2 = make_note(row), make_note(row)
    assert n1 == n2 and len(n1) > 100
    assert "<30" not in n1 and "readmit" not in n1.lower().replace("readmission", "")
    assert with_rag_sentence("note", 0.4).endswith("0.40.")


def test_stratified_split_ratios():
    df = clean(_raw(1000))
    tr, va, te = stratified_split(df)
    assert len(tr) == 700 and len(va) == 150 and len(te) == 150
    for s in (tr, va, te):
        assert abs(s["label"].mean() - df["label"].mean()) < 0.03


def test_features_matrix_shapes():
    df = clean(_raw(200))
    tr, va, _ = stratified_split(df, test_size=0.2, val_size=0.2)
    for name, split, loo in (("tr", tr, True), ("va", va, False)):
        _ = name, loo
    pre = build_preprocessor(tr)
    pre.fit(tr)
    x0 = to_matrix(pre, tr, use_rag=False)
    assert x0.shape[0] == len(tr)
    tr2 = tr.copy()
    tr2[["rag_rate", "rag_mean_sim", "rag_max_sim"]] = 0.1
    x1 = to_matrix(pre, tr2, use_rag=True)
    assert x1.shape == (len(tr), x0.shape[1] + 3)


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    s = np.array([0.1, 0.2, 0.8, 0.7, 0.6, 0.4, 0.9, 0.3])
    m = binary_metrics(y, (s >= 0.5).astype(int), s)
    assert set(m) == {"auc", "f1", "precision", "recall"}
    assert m["auc"] > 0.9
    thr = tune_threshold(y, s)
    assert 0.05 <= thr <= 0.95
