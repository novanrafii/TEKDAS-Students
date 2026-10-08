# Student Retention Dashboard

Dashboard pembelajaran mengikuti pola folder `simulation`, menggunakan dataset `student_retention.csv` di folder ini.

## Menu dashboard

- **Data Mahasiswa** — tabel, filter target, distribusi kelas, dan kualitas data.
- **Business Intelligence** — ringkasan target, usia, nilai, mata kuliah lulus, dan status pembayaran.
- **Machine Learning** — evaluasi Random Forest multiclass, confusion matrix, feature importance, dan simulasi prediksi.
- **LLM Analyst** — tampilkan prompt berbasis profil/prediksi, lalu (opsional) minta analisis ke Ollama lokal.

## Menjalankan aplikasi

```powershell
cd student_doa
pip install -r requirements.txt
streamlit run app.py
```

## Skrip pendukung

Jalankan dari folder `student_doa` (atau gunakan path file dari folder lain):

```powershell
python lihat_data.py
python data_prep.py
python ml_model.py
```

- `lihat_data.py` menampilkan ringkasan dataset dan nilai kosong.
- `data_prep.py` memvalidasi dataset, membuat split stratified, serta menulis `student_train.csv`, `student_test.csv`, `student_ml_dataset.csv`, dan `data_quality_report.json`.
- `ml_model.py` melatih dan mengevaluasi pipeline; artefak disimpan di `artifacts/` dan hasil ringkas diekspor ke CSV/JSON.

## Mengaktifkan LLM

Menu prompt tetap dapat digunakan tanpa Ollama. Untuk menghasilkan jawaban:

1. Pasang dan jalankan Ollama secara terpisah.
2. Jalankan `ollama serve` dan `ollama pull llama3.2`.
3. Jalankan dashboard; model dapat diubah melalui environment variable `OLLAMA_MODEL`.

Uji susunan prompt tanpa koneksi LLM dengan `python llm.py --prompt-only`.

## Catatan interpretasi

Kolom kode seperti `Course` dan kualifikasi orang tua adalah kategori, bukan angka berskala. Model/feature importance menunjukkan pola prediktif, bukan penyebab. Gunakan hasil sebagai bahan eksplorasi dan verifikasi dengan konteks kampus.
