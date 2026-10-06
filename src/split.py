"""Stratified 70/15/15 split. Retrieval index is ALWAYS fit on train only."""
from sklearn.model_selection import train_test_split
from .config import RANDOM_STATE


def stratified_split(df, test_size=0.15, val_size=0.15, seed=RANDOM_STATE):
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"])
    val_frac = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_frac, random_state=seed, stratify=train_val["label"])
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))
