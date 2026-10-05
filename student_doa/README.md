# Student Retention Dashboard

Aplikasi ini dibuat berdasarkan pola dashboard di folder `simulation`, tetapi menggunakan dataset yang tersedia pada folder `student_doa`.

## Dataset

- `student_retention.csv`

Dataset berisi profil mahasiswa dan label target:

- `Dropout`
- `Graduate`
- `Enrolled`

## Menjalankan aplikasi

```bash
cd student_doa
pip install -r requirements.txt
streamlit run app.py
```

## Fitur utama

- Tampilan data mahasiswa
- Distribusi target
- Insight deskriptif usia dan performa akademik
- Evaluasi model machine learning
- Prediksi label untuk record tertentu dari dataset

## Catatan

Aplikasi ini dibuat untuk pembelajaran dan visualisasi serta prediksi retensi mahasiswa dengan pendekatan sederhana menggunakan `scikit-learn`.
