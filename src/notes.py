"""Synthetic clinical-note summaries enabling RAG-style retrieval.

The source dataset has coded fields, not free text. This deterministic template
turns each encounter into a discharge-summary-style string so that semantic
retrieval over 'similar past cases' is meaningful.

Design: value-conditioned clinical phrasing (no label use, no randomness).
High-contrast facts (prior admissions, glycemic control, acuity) are stated in
words so embedding similarity reflects clinical similarity.
"""
import pandas as pd

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "insulin",
]


def _meds_on(row) -> str:
    on = [m for m in MED_COLS if str(row.get(m, "No")).lower() in ("steady", "up", "down")]
    return ", ".join(on) if on else "none"


def _a1c_phrase(v) -> str:
    v = str(v)
    if ">8" in v:
        return "hemoglobin A1C above 8 percent indicating poor glycemic control"
    if ">7" in v:
        return "hemoglobin A1C above 7 percent indicating suboptimal glycemic control"
    if "Norm" in v:
        return "hemoglobin A1C in the normal range"
    return "hemoglobin A1C not measured during this encounter"


def _glu_phrase(v) -> str:
    v = str(v)
    if ">300" in v or ">200" in v:
        return "severely elevated serum glucose on admission testing"
    if "Norm" in v:
        return "normal serum glucose on admission testing"
    return "serum glucose not tested on admission"


def _util_phrase(n_inp, n_er, n_out) -> str:
    bits = []
    try:
        n_inp = int(n_inp)
    except (TypeError, ValueError):
        n_inp = 0
    try:
        n_er = int(n_er)
    except (TypeError, ValueError):
        n_er = 0
    if n_inp >= 3:
        bits.append(f"a high utilizer with {n_inp} inpatient admissions in the past year")
    elif n_inp >= 1:
        bits.append(f"with {n_inp} prior inpatient admission(s) in the past year")
    else:
        bits.append("with no inpatient admissions in the past year")
    if (n_er or 0) >= 2:
        bits.append(f"and {n_er} recent emergency department visits")
    elif (n_er or 0) == 1:
        bits.append("and one recent emergency department visit")
    return " ".join(bits) + "."


def _stay_phrase(los) -> str:
    try:
        los = int(los)
    except (TypeError, ValueError):
        return "Length of stay not recorded."
    if los >= 7:
        return f"Prolonged hospitalization of {los} days."
    if los >= 4:
        return f"Hospital stay of {los} days."
    return f"Brief hospital stay of {los} day(s)."


def make_note(row: pd.Series) -> str:
    change = str(row.get("change", "?"))
    change_phrase = ("diabetes medication regimen was adjusted during the stay"
                     if change == "Ch" else "diabetes medications continued unchanged")
    return (
        f"{row.get('age', '?')} {row.get('gender', '?')} patient of {row.get('race', '?')} race, "
        f"{_util_phrase(row.get('number_inpatient', 0), row.get('number_emergency', 0), row.get('number_outpatient', 0))} "
        f"{_stay_phrase(row.get('time_in_hospital', '?'))} "
        f"Primary diagnosis code {row.get('diag_1', '?')}, secondary codes {row.get('diag_2', '?')} and "
        f"{row.get('diag_3', '?')} ({row.get('number_diagnoses', '?')} diagnoses documented). "
        f"Workup included {row.get('num_lab_procedures', '?')} laboratory tests and "
        f"{row.get('num_procedures', '?')} procedures; {row.get('num_medications', '?')} medications administered. "
        f"Diabetes assessment: {_a1c_phrase(row.get('A1Cresult', '?'))}; {_glu_phrase(row.get('max_glu_serum', '?'))}. "
        f"On {('insulin therapy' if str(row.get('insulin', 'No')).lower() in ('steady', 'up', 'down') else 'oral agents without insulin')}; "
        f"active diabetes agents: {_meds_on(row)}; {change_phrase}. "
        f"Admission type {row.get('admission_type_id', '?')} via source {row.get('admission_source_id', '?')}, "
        f"discharged with disposition {row.get('discharge_disposition_id', '?')}."
    )


def add_notes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["note"] = df.apply(make_note, axis=1)
    return df


def with_rag_sentence(note: str, rag_rate: float) -> str:
    return f"{note} Similar past cases readmission rate: {rag_rate:.2f}."
