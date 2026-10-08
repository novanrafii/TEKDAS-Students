"""Print a quick profile of student_retention.csv.

Run from any working directory with: python lihat_data.py
"""
from pathlib import Path

import pandas as pd

CSV_FILE = Path(__file__).resolve().parent / "student_retention.csv"
df = pd.read_csv(CSV_FILE)

print(f"File: {CSV_FILE.name}")
print(f"Shape: {df.shape[0]:,} baris × {df.shape[1]} kolom")
print("\nTipe data:")
print(df.dtypes.to_string())
print("\nNilai kosong per kolom (terbanyak):")
print(df.isna().sum().sort_values(ascending=False).head(10).to_string())
print("\nDistribusi Target:")
if "Target" in df.columns:
    print(df["Target"].value_counts(dropna=False).to_string())
print("\nContoh 5 baris:")
print(df.head().to_string(index=False))
