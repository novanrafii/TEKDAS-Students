from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Coffee Shop Sales", page_icon="☕", layout="wide")

DATA_FILE = Path(__file__).resolve().parent / "data_prepared" / "coffee_sales_prepared.csv"
REQUIRED_COLUMNS = {
	"transaction_id",
	"transaction_date",
	"transaction_time",
	"transaction_qty",
	"store_location",
	"product_id",
	"product_category",
	"product_detail",
}
COLORS = ["#087F8C", "#7656B5", "#D98B39", "#C65353", "#527A43"]


@st.cache_data(show_spinner="Memuat data penjualan...")
def load_data(path: str, modified_time: int) -> pd.DataFrame:
	"""Muat data hasil audit dan siapkan kolom turunan yang dibutuhkan dashboard."""
	del modified_time  # Dipakai sebagai kunci cache agar file yang berubah dimuat ulang.
	data = pd.read_csv(path)
	missing_columns = REQUIRED_COLUMNS - set(data.columns)
	if missing_columns:
		raise ValueError(
			"Kolom wajib tidak ditemukan pada CSV: "
			+ ", ".join(sorted(missing_columns))
		)

	data["transaction_date"] = pd.to_datetime(
		data["transaction_date"], errors="coerce"
	)
	data["transaction_id"] = pd.to_numeric(
		data["transaction_id"], errors="coerce"
	)
	data["transaction_qty"] = pd.to_numeric(
		data["transaction_qty"], errors="coerce"
	)

	if "sales" not in data.columns:
		if "unit_price" not in data.columns:
			raise ValueError("Kolom sales atau unit_price tidak tersedia pada CSV.")
		data["sales"] = data["transaction_qty"] * pd.to_numeric(
			data["unit_price"], errors="coerce"
		)
	else:
		data["sales"] = pd.to_numeric(data["sales"], errors="coerce")

	# Kolom hour tidak disimpan oleh skrip audit; turunkan dari transaction_time.
	parsed_time = pd.to_datetime(
		data["transaction_time"].astype("string").str.strip(),
		format="%H:%M:%S",
		errors="coerce",
	)
	data["hour"] = parsed_time.dt.hour.astype("Int64")
	data["month"] = data["transaction_date"].dt.to_period("M").astype("string")
	return data


def format_angka(value, decimals: int = 2) -> str:
	"""Format angka dengan pemisah ribuan titik dan desimal koma."""
	if pd.isna(value):
		return "Belum tersedia"
	return (
		f"{value:,.{decimals}f}"
		.replace(",", "_")
		.replace(".", ",")
		.replace("_", ".")
	)


def build_monthly_summary(data: pd.DataFrame, months: list[str]) -> pd.DataFrame:
	"""Buat ringkasan bulanan dan sisipkan bulan tanpa transaksi sebagai kosong."""
	monthly = (
		data.groupby("month")
		.agg(
			total_sales=("sales", "sum"),
			total_transactions=("transaction_id", "nunique"),
			total_units=("transaction_qty", "sum"),
		)
		.reindex(months)
	)
	previous_sales = monthly["total_sales"].shift(1)
	monthly["growth_pct"] = (
		(monthly["total_sales"] - previous_sales) / previous_sales * 100
	).where(previous_sales.notna() & previous_sales.ne(0))
	monthly["average_transaction_value"] = (
		monthly["total_sales"] / monthly["total_transactions"]
	)
	monthly.index.name = "month"
	return monthly.reset_index()


st.title("Dashboard Coffee Shop Sales")
st.caption(
	"Januari–Juni 2023 · Mata uang belum dinyatakan pada file sumber."
)

if not DATA_FILE.is_file():
	st.error(
		"Data hasil audit belum ditemukan. Jalankan "
		"`python audit_prepare_coffee.py` terlebih dahulu."
	)
	st.stop()

try:
	df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime_ns)
except (OSError, pd.errors.ParserError, ValueError) as error:
	st.error(f"Data tidak dapat dimuat: {error}")
	st.stop()

invalid_dates = int(df["transaction_date"].isna().sum())
if invalid_dates:
	st.warning(
		f"{invalid_dates:,} baris memiliki tanggal tidak valid dan tidak dapat "
		"dimasukkan ke filter periode."
	)
	df = df.dropna(subset=["transaction_date"])

invalid_times = int(df["hour"].isna().sum())
if invalid_times:
	st.warning(
		f"{invalid_times:,} baris memiliki waktu tidak valid dan tidak akan "
		"dihitung pada grafik transaksi per jam."
	)

if df.empty:
	st.warning("Tidak ada baris dengan tanggal transaksi yang valid.")
	st.stop()

min_date = df["transaction_date"].min().date()
max_date = df["transaction_date"].max().date()
st.sidebar.header("Filter data")
date_range = st.sidebar.date_input(
	"Rentang tanggal",
	value=(min_date, max_date),
	min_value=min_date,
	max_value=max_date,
	format="DD/MM/YYYY",
)
if isinstance(date_range, (tuple, list)):
	start_date = date_range[0]
	end_date = date_range[-1]
else:
	start_date = end_date = date_range

stores = sorted(df["store_location"].dropna().unique().tolist())
selected_stores = st.sidebar.multiselect(
	"Pilih gerai", stores, default=stores
)
categories = sorted(df["product_category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
	"Pilih kategori", categories, default=categories
)
st.sidebar.caption("Filter diterapkan ke seluruh indikator dan grafik.")

date_mask = df["transaction_date"].dt.date.between(start_date, end_date)
data_filter = df.loc[
	date_mask
	& df["store_location"].isin(selected_stores)
	& df["product_category"].isin(selected_categories)
].copy()

if data_filter.empty:
	st.info("Tidak ada transaksi yang cocok dengan filter yang dipilih.")
	st.stop()

start_month = pd.Period(start_date, freq="M")
end_month = pd.Period(end_date, freq="M")
months = pd.period_range(start_month, end_month, freq="M").astype(str).tolist()
monthly = build_monthly_summary(data_filter, months)
last_month = pd.Period(months[-1], freq="M")

st.caption(
	f"Menampilkan {len(data_filter):,} baris transaksi dari "
	f"{start_date:%d/%m/%Y} sampai {end_date:%d/%m/%Y}."
)
st.subheader("Ringkasan penjualan")
st.caption(
	"Pertumbuhan membandingkan bulan terakhir pada periode terpilih "
	f"({last_month}) dengan bulan sebelumnya."
)

total_sales = data_filter["sales"].sum()
total_transactions = int(data_filter["transaction_id"].nunique())
average_value = total_sales / total_transactions if total_transactions else None
growth = monthly["growth_pct"].iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total penjualan", format_angka(total_sales))
col2.metric("Transaksi tercatat", format_angka(total_transactions, 0))
col3.metric("Rata-rata nilai transaksi", format_angka(average_value))
col4.metric(
	f"Pertumbuhan {last_month} vs {last_month - 1}",
	f"{format_angka(growth)}%" if pd.notna(growth) else "Belum tersedia",
)

st.subheader("Tren penjualan bulanan")
trend = px.line(
	monthly,
	x="month",
	y="total_sales",
	markers=True,
	labels={"month": "Bulan", "total_sales": "Nilai penjualan"},
)
trend.update_traces(line_color=COLORS[0], connectgaps=False)
trend.update_layout(xaxis_type="category", yaxis_rangemode="tozero")
st.plotly_chart(trend, width="stretch", key="trend")

with st.expander("Lihat angka ringkasan bulanan"):
	monthly_table = monthly.rename(
		columns={
			"month": "Bulan",
			"total_sales": "Total penjualan",
			"total_transactions": "Transaksi tercatat",
			"total_units": "Unit terjual",
			"growth_pct": "Pertumbuhan (%)",
			"average_transaction_value": "Rata-rata nilai transaksi",
		}
	)
	st.dataframe(
		monthly_table.style.format(
			{
				"Total penjualan": "{:,.2f}",
				"Transaksi tercatat": "{:,.0f}",
				"Unit terjual": "{:,.0f}",
				"Pertumbuhan (%)": "{:,.2f}",
				"Rata-rata nilai transaksi": "{:,.2f}",
			},
			na_rep="Belum tersedia",
			decimal=",",
			thousands=".",
		),
		width="stretch",
		hide_index=True,
	)

st.divider()
st.header("Perbandingan gerai dan kategori")
left, right = st.columns(2)

with left:
	st.subheader("Penjualan per gerai")
	sales_by_store = (
		data_filter.groupby("store_location", as_index=False)
		.agg(total_sales=("sales", "sum"))
		.sort_values("total_sales", ascending=False)
	)
	store_chart = px.bar(
		sales_by_store,
		x="store_location",
		y="total_sales",
		color="store_location",
		text_auto=".2s",
		labels={"store_location": "Gerai", "total_sales": "Nilai penjualan"},
		color_discrete_sequence=COLORS,
	)
	store_chart.update_layout(showlegend=False, height=400)
	st.plotly_chart(store_chart, width="stretch", key="stores")

with right:
	st.subheader("Kontribusi penjualan kategori")
	sales_by_category = (
		data_filter.groupby("product_category", as_index=False)
		.agg(total_sales=("sales", "sum"))
		.sort_values("total_sales", ascending=False)
	)
	category_chart = px.pie(
		sales_by_category,
		names="product_category",
		values="total_sales",
		hole=0.55,
		color_discrete_sequence=COLORS,
	)
	category_chart.update_traces(
		textinfo="none",
		hovertemplate=(
			"%{label}<br>Penjualan: %{value:,.2f}<br>"
			"Kontribusi: %{percent}<extra></extra>"
		),
	)
	category_chart.update_layout(height=400, legend_title_text="Kategori")
	st.plotly_chart(category_chart, width="stretch", key="categories")

st.header("Produk dan pola waktu transaksi")
left, right = st.columns(2)

with left:
	st.subheader("10 produk dengan penjualan tertinggi")
	top_products = (
		data_filter.groupby(["product_id", "product_detail"], as_index=False)
		.agg(total_sales=("sales", "sum"))
		.nlargest(10, "total_sales")
		.sort_values("total_sales", ascending=True)
	)
	top_products["label"] = (
		top_products["product_detail"].astype("string")
		+ " ("
		+ top_products["product_id"].astype("string")
		+ ")"
	)
	product_chart = px.bar(
		top_products,
		x="total_sales",
		y="label",
		orientation="h",
		color_discrete_sequence=[COLORS[1]],
		labels={"total_sales": "Nilai penjualan", "label": "Produk"},
	)
	product_chart.update_layout(height=470)
	st.plotly_chart(product_chart, width="stretch", key="products")
	st.caption("Peringkat berdasarkan nilai penjualan, bukan unit atau keuntungan.")

with right:
	st.subheader("Transaksi tercatat menurut jam")
	by_hour = (
		data_filter.dropna(subset=["hour"])
		.groupby("hour", as_index=False)
		.agg(total_transactions=("transaction_id", "nunique"))
		.sort_values("hour")
	)
	if by_hour.empty:
		st.info("Tidak ada waktu transaksi valid untuk ditampilkan.")
	else:
		by_hour["label"] = (
			by_hour["hour"].astype(int).astype(str).str.zfill(2) + ":00"
		)
		hour_chart = px.bar(
			by_hour,
			x="label",
			y="total_transactions",
			color_discrete_sequence=[COLORS[2]],
			labels={"label": "Jam", "total_transactions": "Transaksi tercatat"},
		)
		hour_chart.update_layout(height=470, xaxis_type="category")
		st.plotly_chart(hour_chart, width="stretch", key="hours")
	st.caption("Akumulasi sepanjang periode; bukan ukuran antrean layanan.")

st.download_button(
	"Unduh data sesuai filter (CSV)",
	data=data_filter.to_csv(index=False).encode("utf-8-sig"),
	file_name="coffee_sales_filtered.csv",
	mime="text/csv",
)
st.caption(
	"Transaksi tercatat adalah transaction_id unik, bukan pelanggan unik. "
	"Rata-rata nilai transaksi belum dipastikan sebagai seluruh isi struk pelanggan."
)