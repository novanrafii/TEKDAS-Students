"""Train and evaluate a multiclass student-retention classifier.

Run data_prep.py first, then: python ml_model.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import CATEGORICAL_COLUMNS, ROW_ID, TARGET, load_data

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FEATURE_COLUMNS: list[str] = []


def build_pipeline(feature_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    categorical = [c for c in categorical_columns if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))
    preprocessor = ColumnTransformer(transformers=transformers)
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def evaluate_model(train: pd.DataFrame, test: pd.DataFrame):
    """Fit one pipeline and return artifacts used by both CLI and dashboard."""
    excluded = {TARGET, ROW_ID}
    feature_columns = [c for c in train.columns if c not in excluded]
    X_train, y_train = train[feature_columns], train[TARGET]
    X_test, y_test = test[feature_columns], test[TARGET]
    pipeline = build_pipeline(feature_columns, CATEGORICAL_COLUMNS)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    classes = pipeline.named_steps["model"].classes_

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "weighted_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_weighted": float(roc_auc_score(
            y_test, y_prob, labels=classes, multi_class="ovr", average="weighted"
        )),
        "classes": [str(c) for c in classes],
        "classification_report": classification_report(y_test, y_pred, digits=3, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
    }

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({
        "feature": names,
        "importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return pipeline, metrics, importance, y_pred, y_prob


def fit_from_dataframe(df: pd.DataFrame):
    """Create a reproducible stratified split for dashboard use."""
    from sklearn.model_selection import train_test_split

    data = df.copy().reset_index(drop=True)
    data.insert(0, ROW_ID, range(1, len(data) + 1))
    train, test = train_test_split(
        data, test_size=0.25, random_state=42, stratify=data[TARGET]
    )
    return evaluate_model(train.reset_index(drop=True), test.reset_index(drop=True))


def main() -> None:
    train_path, test_path = BASE_DIR / "student_train.csv", BASE_DIR / "student_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Jalankan 'python data_prep.py' terlebih dahulu.")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    pipeline, metrics, importance, y_pred, y_prob = evaluate_model(train, test)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR / "student_retention_pipeline.joblib")
    importance.to_csv(BASE_DIR / "feature_importance.csv", index=False)
    (BASE_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    prediction = test[[ROW_ID, TARGET]].copy()
    prediction["predicted_target"] = y_pred
    for index, label in enumerate(pipeline.named_steps["model"].classes_):
        prediction[f"probability_{label}"] = y_prob[:, index]
    prediction.to_csv(BASE_DIR / "student_predictions.csv", index=False)

    print("Model Random Forest selesai dilatih.")
    print(metrics["classification_report"])
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"ROC-AUC OVR weighted: {metrics['roc_auc_ovr_weighted']:.3f}")
    print("Disimpan di artifacts/ dan file metrik/prediksi student_doa.")


if __name__ == "__main__":
    main()
