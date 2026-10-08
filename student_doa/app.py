from pathlib import Path

import pandas as pd
import streamlit as st

from data_prep import TARGET, load_data
from llm import DEFAULT_MODEL, build_prompt, call_ollama
from ml_model import fit_from_dataframe

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Student Retention | BI + ML + LLM", layout="wide")
st.title("Student Retention Dashboard")
st.caption("Dashboard pembelajaran untuk memahami data mahasiswa, mengevaluasi prediksi, dan menyusun tindak lanjut berbasis bukti.")


@st.cache_data
def load_dashboard_data():
    return load_data()


@st.cache_resource
def load_dashboard_model(data):
    return fit_from_dataframe(data)


try:
    df = load_dashboard_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Dataset tidak dapat dibaca: {exc}")
    st.stop()

with st.spinner("Menyiapkan model Machine Learning..."):
    model, metrics, feature_importance, y_pred, y_prob = load_dashboard_model(df)

classes = model.named_steps["model"].classes_
st.sidebar.header("Pilih mahasiswa")
selected_row = st.sidebar.selectbox(
    "ID Mahasiswa",
    df.index.tolist(),
    format_func=lambda idx: f"M{idx + 1:05d}",
)
dropout_threshold = st.sidebar.slider(
    "Ambang risiko dropout",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Probabilitas Dropout di atas ambang akan ditandai sebagai prediksi Dropout.",
)
st.sidebar.markdown("---")
st.sidebar.caption("Sumber: student_retention.csv")

dropout_rate = df[TARGET].eq("Dropout").mean() * 100
graduate_rate = df[TARGET].eq("Graduate").mean() * 100
enrolled_rate = df[TARGET].eq("Enrolled").mean() * 100
metric_cols = st.columns(4)
metric_cols[0].metric("Total mahasiswa", f"{len(df):,}")
metric_cols[1].metric("Dropout", f"{int(df[TARGET].eq('Dropout').sum()):,}", f"{dropout_rate:.1f}% dari data")
metric_cols[2].metric("Graduate", f"{int(df[TARGET].eq('Graduate').sum()):,}", f"{graduate_rate:.1f}% dari data")
metric_cols[3].metric("Enrolled", f"{int(df[TARGET].eq('Enrolled').sum()):,}", f"{enrolled_rate:.1f}% dari data")
st.markdown("---")

menu = st.radio(
    "Navigasi utama",
    [
        "1 · Data Mahasiswa",
        "2 · Business Intelligence",
        "3 · Machine Learning",
        "4 · LLM Analyst",
    ],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("---")

if menu == "1 · Data Mahasiswa":
    st.header("Data mahasiswa")
    st.write(f"Dataset memiliki **{df.shape[0]:,} baris** dan **{df.shape[1]} kolom**.")
    selected_target = st.selectbox("Filter status", ["Semua"] + sorted(df[TARGET].dropna().unique().tolist()))
    filtered = df if selected_target == "Semua" else df[df[TARGET].eq(selected_target)]
    st.dataframe(filtered.head(500), width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("Distribusi Target")
        st.bar_chart(df[TARGET].value_counts().rename_axis("Status").sort_index())
    with right:
        st.subheader("Kualitas data")
        quality = pd.DataFrame({
            "Nilai kosong": df.isna().sum(),
            "Persentase kosong (%)": (df.isna().mean() * 100).round(2),
        }).sort_values("Nilai kosong", ascending=False)
        st.dataframe(quality.head(12), width="stretch")
    st.info("Sebagian kolom adalah kode kategori. Arti angka kategori perlu dibaca bersama kamus data; angka tersebut bukan skala nilai.")

elif menu == "2 · Business Intelligence":
    st.header("Business Intelligence")
    st.caption("BI menjelaskan pola historis pada dataset; ringkasan ini tidak menyatakan hubungan sebab-akibat.")
    target_counts = df[TARGET].value_counts().rename_axis("Status").sort_index()
    st.subheader("Komposisi status mahasiswa")
    st.bar_chart(target_counts)

    grade_columns = [
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
    ]
    approved_columns = [
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
    ]
    summary_columns = grade_columns + approved_columns
    summary = df.groupby(TARGET)[summary_columns].mean().round(2)
    st.subheader("Rata-rata nilai dan mata kuliah lulus menurut status")
    st.dataframe(summary, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Usia saat pendaftaran menurut status")
        st.bar_chart(df.groupby(TARGET)["Age at enrollment"].median().sort_values())
    with right:
        st.subheader("Persentase biaya kuliah terbarui")
        tuition = (df.groupby(TARGET)["Tuition fees up to date"].mean() * 100).round(1)
        st.bar_chart(tuition)

    st.subheader("Status utang menurut target")
    debtor_rate = (df.groupby(TARGET)["Debtor"].mean() * 100).round(1).rename("Mahasiswa berstatus debtor (%)")
    st.dataframe(debtor_rate, width="stretch")

elif menu == "3 · Machine Learning":
    st.header("Machine Learning")
    st.caption("Random Forest multiclass dengan split stratified 75/25. Target tidak dipakai sebagai fitur.")
    score_cols = st.columns(4)
    score_cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    score_cols[1].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    score_cols[2].metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")
    score_cols[3].metric("ROC-AUC OVR", f"{metrics['roc_auc_ovr_weighted']:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Classification report")
        st.code(metrics["classification_report"], language="text")
    with right:
        st.subheader("Confusion matrix")
        confusion = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
        st.dataframe(confusion, width="stretch")
    st.subheader("Faktor penting global")
    st.caption("Feature importance menunjukkan kontribusi prediktif global, bukan penyebab status mahasiswa.")
    st.bar_chart(feature_importance.head(15).set_index("feature")["importance"])

    st.subheader("Simulasi prediksi record dataset")
    st.caption(f"Record terpilih: M{selected_row + 1:05d} · ambang risiko dropout: {dropout_threshold:.0%}")
    profile = df.loc[[selected_row]].drop(columns=[TARGET])
    probabilities = model.predict_proba(profile)[0]
    prob_dict = {str(label): float(prob) for label, prob in zip(classes, probabilities)}
    non_dropout_classes = [str(label) for label in classes if str(label) != "Dropout"]
    predicted = (
        "Dropout"
        if prob_dict.get("Dropout", 0.0) >= dropout_threshold
        else max(non_dropout_classes, key=lambda label: prob_dict[label])
    )
    actual = str(df.loc[selected_row, TARGET])
    result_cols = st.columns(3)
    result_cols[0].metric("Label aktual", actual)
    result_cols[1].metric("Prediksi berdasar ambang", predicted)
    result_cols[2].metric("Probabilitas dropout", f"{prob_dict.get('Dropout', 0.0):.1%}")
    st.bar_chart(pd.Series(prob_dict, name="Probabilitas").sort_values(ascending=False))

elif menu == "4 · LLM Analyst":
    st.header("LLM Analyst")
    st.caption(f"Model lokal melalui Ollama: `{DEFAULT_MODEL}`. Prompt dapat ditinjau tanpa menghubungi LLM.")
    st.caption(f"Mahasiswa terpilih dari sidebar: M{selected_row + 1:05d} · ambang dropout: {dropout_threshold:.0%}")
    profile = df.loc[[selected_row]].drop(columns=[TARGET])
    probabilities = model.predict_proba(profile)[0]
    prob_dict = {str(label): float(prob) for label, prob in zip(classes, probabilities)}
    non_dropout_classes = [str(label) for label in classes if str(label) != "Dropout"]
    predicted = (
        "Dropout"
        if prob_dict.get("Dropout", 0.0) >= dropout_threshold
        else max(non_dropout_classes, key=lambda label: prob_dict[label])
    )
    factors = feature_importance.head(8).to_dict("records")
    prompt = build_prompt(
        student_profile=df.loc[selected_row].to_dict(),
        predicted_target=predicted,
        probabilities=prob_dict,
        model_factors=factors,
    )
    st.metric("Prediksi model", predicted, help=f"Probabilitas: {prob_dict[predicted]:.1%}")
    with st.expander("Lihat prompt yang dikirim ke LLM", expanded=True):
        st.code(prompt, language="text")
    if st.button("Minta analisis dari Ollama", type="primary"):
        with st.spinner("Meminta analisis berbasis bukti kepada Ollama..."):
            try:
                st.markdown(call_ollama(prompt))
            except Exception as exc:
                st.error(f"Ollama belum dapat digunakan: {exc}")
                st.info("Pastikan Ollama aktif (`ollama serve`) dan model tersedia (`ollama pull llama3.2`).")

st.markdown("---")
st.caption("Gunakan hasil sebagai bahan eksplorasi; keputusan dukungan mahasiswa perlu verifikasi manusia dan konteks kampus.")
