from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_retention.csv"

st.set_page_config(page_title="Student Retention Dashboard", layout="wide")
st.title("Student Retention Dashboard")
st.caption("Analisis data mahasiswa, insight bisnis, dan prediksi status retensi berbasis ML.")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


df = load_data()
TARGET_COLUMN = "Target"


@st.cache_data
def train_model(df_input):
    X = df_input.drop(columns=[TARGET_COLUMN])
    y = df_input[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="weighted",
        ),
        "report": classification_report(y_test, y_pred, digits=3),
    }

    return model, metrics, X_test, y_test, y_pred


model, metrics, X_test, y_test, y_pred = train_model(df)

if "Target" in df.columns:
    target_counts = df[TARGET_COLUMN].value_counts().sort_index()

    total_students = len(df)
    dropout_rate = (df[TARGET_COLUMN].eq("Dropout").mean() * 100)
    graduate_rate = (df[TARGET_COLUMN].eq("Graduate").mean() * 100)
    enrolled_rate = (df[TARGET_COLUMN].eq("Enrolled").mean() * 100)

    st.sidebar.header("Filter")
    selected_target = st.sidebar.selectbox(
        "Pilih label target",
        ["Semua", "Dropout", "Graduate", "Enrolled"],
    )
    filtered_df = df if selected_target == "Semua" else df[df[TARGET_COLUMN] == selected_target]

    st.sidebar.write(f"Jumlah data terlihat: {len(filtered_df):,}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total mahasiswa", f"{total_students:,}")
    col2.metric("Dropout", f"{dropout_rate:.1f}%")
    col3.metric("Graduate", f"{graduate_rate:.1f}%")
    col4.metric("Enrolled", f"{enrolled_rate:.1f}%")

    tab_data, tab_insight, tab_ml = st.tabs(["1 · Data", "2 · Insight", "3 · Prediksi ML"])

    with tab_data:
        st.subheader("Dataset mahasiswa")
        st.dataframe(filtered_df.head(200), use_container_width=True)

        st.subheader("Ringkasan statistik numerik")
        st.dataframe(df.describe().T, use_container_width=True)

    with tab_insight:
        st.subheader("Distribusi target")
        st.bar_chart(target_counts)

        st.subheader("Distribusi usia saat masuk")
        age_chart = df["Age at enrollment"].value_counts().sort_index()
        st.bar_chart(age_chart)

        st.subheader("Rata-rata nilai per semester berdasarkan target")
        value_cols = [
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
        ]
        sem_summary = df.groupby(TARGET_COLUMN)[value_cols].mean().round(2)
        st.dataframe(sem_summary, use_container_width=True)

    with tab_ml:
        st.subheader("Evaluasi model")
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col_b.metric("ROC-AUC (weighted)", f"{metrics['roc_auc']:.3f}")
        st.code(metrics["report"], language="text")

        st.subheader("Prediksi untuk mahasiswa tertentu")
        selected_row = st.selectbox("Pilih baris data", df.index.tolist(), format_func=lambda idx: f"Mahasiswa #{idx} - {df.loc[idx, TARGET_COLUMN]}")
        student = df.loc[[selected_row]].reset_index(drop=True)
        prediction = model.predict(student.drop(columns=[TARGET_COLUMN]))[0]
        probabilities = model.predict_proba(student.drop(columns=[TARGET_COLUMN]))[0]
        prob_dict = {label: float(prob) for label, prob in zip(model.named_steps["classifier"].classes_, probabilities)}

        actual_label = student[TARGET_COLUMN].iloc[0]
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Label aktual", actual_label)
        result_col2.metric("Prediksi model", prediction)
        result_col3.metric("Probabilitas tertinggi", f"{max(prob_dict.values()):.1%}")

        st.write("Probabilitas per kelas:")
        st.bar_chart(pd.Series(prob_dict).sort_values(ascending=False))

        st.subheader("Profil mahasiswa yang dipilih")
        st.dataframe(student, use_container_width=True)

else:
    st.error("Kolom target 'Target' tidak ditemukan pada dataset.")

st.markdown("---")
st.caption("Dashboard ini memanfaatkan data yang ada pada folder student_doa dan menyajikan analisis serta prediksi retensi mahasiswa.")
