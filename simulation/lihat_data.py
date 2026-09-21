"""Menampilkan ringkasan dari customers.csv.

Jalankan dari folder simulation:
    python lihat_data.py

Atau dari root project:
    python simulation/lihat_data.py
"""
from pathlib import Path

import pandas as pd


# Path dibuat berdasarkan lokasi file ini, sehingga tidak bergantung pada
# folder terminal yang sedang aktif.
csv_file = Path(__file__).resolve().parent / "customers.csv"
df = pd.read_csv(csv_file)

print(df.shape)
print(df.dtypes)
print(df.head())
print(df.isna().mean().sort_values(ascending=False).head())
