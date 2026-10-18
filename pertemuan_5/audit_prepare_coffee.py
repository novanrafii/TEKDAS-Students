"""Audit dan persiapan Coffee Shop Sales untuk praktik BI.

File Excel sumber tidak diubah dan transaksi tidak dihapus otomatis.
Jalankan: python audit_prepare_coffee.py
Opsional: python audit_prepare_coffee.py --input "lokasi/file.xlsx"
"""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_FOLDER = Path(__file__).resolve().parent
SHEET_NAME = "Transactions"
COLUMNS = [
   "transaction_id",
   "transaction_date",
   "transaction_time",
   "transaction_qty",
   "store_id",
   "store_location",
   "product_id",
   "unit_price",
   "product_category",
   "product_type",
   "product_detail",
]
TEXT_COLUMNS = [
   "store_location",
   "product_category",
   "product_type",
   "product_detail",
]
NUMERIC_COLUMNS = [
   "transaction_id",
   "transaction_qty",
   "store_id",
   "product_id",
   "unit_price",
]
INTEGER_COLUMNS = {"transaction_id", "transaction_qty", "store_id", "product_id"}
DIMENSION_COLUMNS = {
   "store_id": ["store_location"],
   "product_id": ["product_category", "product_type", "product_detail"],
}


def _json_number(value):
   """Convert a numeric result to a JSON-safe float or None."""
   if pd.isna(value) or not float("-inf") < float(value) < float("inf"):
      return None
   return float(value)


def main():
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--input",
      type=Path,
      default=DEFAULT_FOLDER / "Coffee Shop Sales.xlsx",
      help="Lokasi file Excel sumber.",
   )
   parser.add_argument(
      "--output",
      type=Path,
      default=None,
      help="Folder keluaran (default: data_prepared di samping file sumber).",
   )
   args = parser.parse_args()

   source = args.input.resolve()
   if not source.is_file():
      raise FileNotFoundError(f"File sumber tidak ditemukan: {source}")

   output = (args.output or source.parent / "data_prepared").resolve()
   output.mkdir(parents=True, exist_ok=True)

   df = pd.read_excel(source, sheet_name=SHEET_NAME)
   missing_columns = sorted(set(COLUMNS) - set(df.columns))
   if missing_columns:
      raise ValueError(f"Kolom wajib tidak tersedia: {missing_columns}")
   if df.empty:
      raise ValueError("Dataset tidak memiliki baris.")

   issues = []
   report = {
      "source_file": source.name,
      "sheet": SHEET_NAME,
      "original_rows": int(len(df)),
      "original_columns": list(df.columns),
      "missing_before": {
         column: int(count) for column, count in df.isna().sum().items()
      },
      "duplicate_rows_before": int(df.duplicated().sum()),
      "trimmed_values": {},
      "dimension_mapping_conflicts": {},
   }

   for column in TEXT_COLUMNS:
      text = df[column].astype("string")
      cleaned = text.str.strip()
      report["trimmed_values"][column] = int(
         text.ne(cleaned).fillna(False).sum()
      )
      df[column] = cleaned
      if df[column].isna().any() or df[column].eq("").any():
         issues.append(f"{column}: teks kosong atau hilang")

   for column in NUMERIC_COLUMNS:
      df[column] = pd.to_numeric(df[column], errors="coerce")
      values = df[column].dropna()
      if df[column].isna().any():
         issues.append(f"{column}: angka tidak valid atau hilang")
      if (values <= 0).any():
         issues.append(f"{column}: angka nol atau negatif")
      if not values.map(lambda value: float("-inf") < value < float("inf")).all():
         issues.append(f"{column}: angka tak hingga")
      if column in INTEGER_COLUMNS and (values % 1 != 0).any():
         issues.append(f"{column}: seharusnya bilangan bulat")

   df["transaction_date"] = pd.to_datetime(
      df["transaction_date"], errors="coerce"
   )
   if df["transaction_date"].isna().any():
      issues.append("transaction_date: tanggal tidak valid atau hilang")
   elif df["transaction_date"].ne(df["transaction_date"].dt.normalize()).any():
      issues.append(
         "transaction_date mengandung jam. Periksa sebelum menggabungkan waktu."
      )

   time_text = df["transaction_time"].astype("string").str.strip()
   parsed_time = pd.to_datetime(
      time_text, format="%H:%M:%S", errors="coerce"
   )
   if parsed_time.isna().any():
      issues.append("transaction_time: waktu tidak valid, harus HH:MM:SS")
   df["transaction_time"] = parsed_time.dt.strftime("%H:%M:%S").astype("string")

   report["duplicate_transaction_ids"] = int(
      df["transaction_id"].duplicated().sum()
   )
   report["duplicate_rows_after_trim"] = int(df.duplicated().sum())
   if report["duplicate_transaction_ids"]:
      issues.append("transaction_id berulang: pastikan grain sebelum lanjut")
   if report["duplicate_rows_after_trim"]:
      issues.append("Ada baris duplikat, perlu diperiksa sebelum lanjut")

   for key, attributes in DIMENSION_COLUMNS.items():
      conflicts = {}
      valid_rows = df.dropna(subset=[key])
      for attribute in attributes:
         counts = valid_rows.groupby(key, dropna=True)[attribute].nunique(
            dropna=True
         )
         conflict_ids = counts[counts > 1].index.tolist()
         if conflict_ids:
            conflicts[attribute] = [
               value.item() if hasattr(value, "item") else value
               for value in conflict_ids
            ]
      report["dimension_mapping_conflicts"][key] = conflicts
      if conflicts:
         issues.append(f"{key}: nilai atribut berbeda untuk ID yang sama")

   df["sales"] = df["transaction_qty"] * df["unit_price"]
   df["month"] = df["transaction_date"].dt.to_period("M").astype("string")

   monthly = (
      df.groupby("month", as_index=False, dropna=True)
      .agg(
         total_sales=("sales", "sum"),
         total_transactions=("transaction_id", "nunique"),
         total_units=("transaction_qty", "sum"),
      )
      .sort_values("month")
      .reset_index(drop=True)
   )
   previous_sales = monthly["total_sales"].shift(1)
   periods = pd.PeriodIndex(monthly["month"], freq="M")
   consecutive_months = pd.Series(periods.asi8).diff().eq(1)
   monthly["growth_pct"] = (
      (monthly["total_sales"] - previous_sales) / previous_sales * 100
   ).where(previous_sales.ne(0) & consecutive_months)
   monthly["average_transaction_value"] = (
      monthly["total_sales"] / monthly["total_transactions"]
   )

   transaction_count = int(df["transaction_id"].nunique())
   total_sales = df["sales"].sum(min_count=1)
   total_units = df["transaction_qty"].sum(min_count=1)
   report.update(
      {
         "total_sales": _json_number(total_sales),
         "total_units": (
            int(total_units) if pd.notna(total_units) else None
         ),
         "total_transactions": transaction_count,
         "average_recorded_transaction_value": (
            _json_number(total_sales / transaction_count)
            if transaction_count and pd.notna(total_sales)
            else None
         ),
         "currency": "Tidak dinyatakan pada file sumber",
         "grain": (
            "Satu baris per transaction_id, satu produk. Belum dipastikan "
            "sebagai seluruh isi struk."
         ),
         "issues": issues,
         "audit_status": "needs_review" if issues else "passed",
         "prepared_rows": int(len(df)),
         "rows_removed": 0,
      }
   )

   df.to_csv(
      output / "coffee_sales_prepared.csv",
      index=False,
      encoding="utf-8-sig",
      date_format="%Y-%m-%d %H:%M:%S",
   )
   monthly.to_csv(output / "monthly_summary.csv", index=False, encoding="utf-8-sig")
   (output / "audit_report.json").write_text(
      json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
      encoding="utf-8",
   )

   print(f"\nSTATUS AUDIT: {report['audit_status'].upper()}")
   print("Perbaikan spasi:", report["trimmed_values"])
   print(f"Baris dipertahankan: {len(df):,}")
   print(f"Total penjualan: {report['total_sales']:,.2f}")
   print(f"Total unit: {report['total_units']:,}")
   if issues:
      print("\nPerlu ditinjau:")
      for issue in issues:
         print(f"- {issue}")
   print("\nRingkasan bulanan:\n", monthly.round(2).to_string(index=False))
   print("\nOutput:", output)


if __name__ == "__main__":
   main()