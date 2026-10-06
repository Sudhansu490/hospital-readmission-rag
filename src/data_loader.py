"""Load + clean the Diabetes 130-US hospitals dataset."""
import pandas as pd
from .config import DATA_CSV, DROP_COLS, LABEL_POS


def load_raw(csv_path=DATA_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path} — place diabetic_data.csv in data/")
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # binary label: 30-day readmission
    df["label"] = (df["readmitted"] == LABEL_POS).astype(int)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop(columns=["readmitted"])
    # '?' -> 'Missing' for categoricals (keep signal instead of dropping rows).
    # is_string_dtype covers both pandas-2 object and pandas-3 str dtypes.
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].replace("?", "Missing")
    return df
