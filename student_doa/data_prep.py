"""Validate and split the student retention dataset for ML.

Run from any working directory with: python data_prep.py
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_retention.csv"
TARGET = "Target"
ROW_ID = "student_row_id"

# Integer codes represent categories in the source dataset, not measurements.
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the source CSV and validate its required target column."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan di {path.name}.")
    if df[TARGET].isna().any():
        raise ValueError("Kolom Target memiliki nilai kosong; bersihkan label sebelum training.")
    if df[TARGET].nunique() < 2:
        raise ValueError("Model klasifikasi memerlukan setidaknya dua kelas Target.")
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    """Add traceable row IDs and create a stratified train/test split."""
    prepared = df.copy().reset_index(drop=True)
    prepared.insert(0, ROW_ID, range(1, len(prepared) + 1))
    train, test = train_test_split(
        prepared,
        test_size=test_size,
        random_state=random_state,
        stratify=prepared[TARGET],
    )
    return prepared, train.reset_index(drop=True), test.reset_index(drop=True)


def build_quality_report(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Summarize data quality and class balance without silently changing data."""
    return {
        "source_file": DATA_FILE.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target": TARGET,
        "target_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "categorical_code_columns": [c for c in CATEGORICAL_COLUMNS if c in df.columns],
        "note": "Kode kategori diperlakukan sebagai kategori; data tidak dihapus otomatis selain tidak ada perubahan baris.",
    }


def main() -> None:
    df = load_data()
    prepared, train, test = prepare_data(df)
    prepared.to_csv(BASE_DIR / "student_ml_dataset.csv", index=False)
    train.to_csv(BASE_DIR / "student_train.csv", index=False)
    test.to_csv(BASE_DIR / "student_test.csv", index=False)
    report = build_quality_report(df, train, test)
    (BASE_DIR / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Persiapan data selesai.")
    print(f"Shape sumber : {df.shape[0]:,} baris × {df.shape[1]} kolom")
    print(f"Train / test: {len(train):,} / {len(test):,}")
    print("Distribusi target:")
    print(df[TARGET].value_counts().to_string())
    print("Dibuat: student_ml_dataset.csv, student_train.csv, student_test.csv")
    print("Dibuat: data_quality_report.json")


if __name__ == "__main__":
    main()
