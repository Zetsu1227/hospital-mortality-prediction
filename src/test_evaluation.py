"""Refit the preselected pipeline on all labels and score an official test CSV."""

from __future__ import annotations

import json
import hashlib
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


TARGET = "desfecho"
ID_COLUMN = "id"
BINARY_FEATURES = (
    [f"dadoclin{i}" for i in range(1, 15)]
    + ["sociodemografico1", "sociodemografico2"]
    + [f"comorbidade{i}" for i in range(1, 8)]
)
NUMERIC_FEATURES = ["sociodemografico3"]
NOMINAL_FEATURES = ["sociodemografico4", "sociodemografico5"]
PREDICTOR_COLUMNS = BINARY_FEATURES + NUMERIC_FEATURES + NOMINAL_FEATURES
MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "svc_rbf",
]


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a delimited file while detecting comma, semicolon, or tab separators."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error as exc:
        raise ValueError(f"Não foi possível detectar o delimitador de {path.name}.") from exc
    return pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")


def _normalise_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.str.strip().str.lower()
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Colunas repetidas após normalização em {path.name}: {duplicates}")
    return frame


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("binary", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "nominal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                NOMINAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def _build_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            solver="lbfgs", l1_ratio=0.0, max_iter=3000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400, random_state=42, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=42, early_stopping=False
        ),
        "svc_rbf": SVC(kernel="rbf"),
    }


def _validate_inputs(
    train: pd.DataFrame, test: pd.DataFrame, train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series | None]:
    required_train = {ID_COLUMN, TARGET, *PREDICTOR_COLUMNS}
    missing_train = required_train - set(train.columns)
    if missing_train:
        raise ValueError(f"Colunas ausentes no treino: {sorted(missing_train)}")

    required_test = {ID_COLUMN, *PREDICTOR_COLUMNS}
    missing_test = required_test - set(test.columns)
    extra_test = set(test.columns) - (required_test | {TARGET})
    if missing_test or extra_test:
        raise ValueError(
            "Esquema do teste incompatível. "
            f"Ausentes: {sorted(missing_test)}; extras: {sorted(extra_test)}. "
            "O teste deve ter ID, os mesmos preditores do treino e, opcionalmente, DESFECHO."
        )

    for frame, path in ((train, train_path), (test, test_path)):
        if frame[ID_COLUMN].isna().any():
            raise ValueError(f"Há IDs ausentes em {path.name}.")
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"Há IDs duplicados em {path.name}.")

    y_train = train[TARGET]
    if y_train.isna().any() or not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("O alvo do treino deve estar completo e conter somente 0 e 1.")
    if y_train.nunique() != 2:
        raise ValueError("O treino precisa conter as duas classes do desfecho.")

    for column in BINARY_FEATURES:
        observed = set(train[column].dropna().unique())
        if not observed.issubset({0, 1}):
            raise ValueError(f"Valores inesperados em variável binária {column}: {observed}")

    y_test = test[TARGET] if TARGET in test.columns else None
    if y_test is not None:
        if y_test.isna().any() or not set(y_test.unique()).issubset({0, 1}):
            raise ValueError("O alvo do teste, quando presente, deve conter somente 0 e 1.")

    return (
        train[PREDICTOR_COLUMNS].copy(),
        y_train.astype(int).copy(),
        test[PREDICTOR_COLUMNS].copy(),
        None if y_test is None else y_test.astype(int).copy(),
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_official_test(
    project_root: str | Path,
    train_filename: str = "final_treino_val.csv",
    test_filename: str = "final_teste.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure | None, dict[str, Any]]:
    """Train all saved candidates on all labeled training rows and score the test.

    The test file may omit ``desfecho`` (normal competition test case). If it
    contains labels, those labels are used only after fitting to compute metrics;
    they never enter preprocessing, hyperparameter selection, or threshold choice.
    """
    root = Path(project_root).resolve()
    raw_dir = root / "data" / "raw"
    train_path = raw_dir / train_filename
    test_path = raw_dir / test_filename
    search_path = root / "results" / "stage4_model_search.json"
    for required_path in (train_path, test_path, search_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Arquivo necessário não encontrado: {required_path}. "
                "Coloque o teste como data/raw/final_teste.csv e execute antes a busca final no notebook 01."
            )

    train = _normalise_columns(_read_csv(train_path), train_path)
    test = _normalise_columns(_read_csv(test_path), test_path)
    X_train, y_train, X_test, y_test = _validate_inputs(
        train, test, train_path, test_path
    )

    search_results = json.loads(search_path.read_text(encoding="utf-8"))
    model_params = search_results["best_params_on_all_labeled_data"]
    absent_models = set(MODEL_NAMES) - set(model_params)
    if absent_models:
        raise ValueError(f"Artefatos incompletos para os modelos: {sorted(absent_models)}")

    # Se os dados rotulados mudaram, os parâmetros salvos podem estar obsoletos.
    protocol = search_results.get("protocol", {})
    expected_hash = protocol.get("labeled_training_file_sha256")
    if expected_hash is not None:
        actual_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                "O CSV de treino mudou desde a busca da etapa 4. "
                "Execute novamente o notebook 01 antes de pontuar o teste."
            )

    preprocessor = _build_preprocessor()
    models = _build_models()
    score_columns: dict[str, np.ndarray] = {}
    score_types: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for name in MODEL_NAMES:
        pipeline = Pipeline(
            steps=[("preprocessor", clone(preprocessor)), ("model", clone(models[name]))]
        )
        pipeline.set_params(**model_params[name]["best_params"])
        # Somente o CSV de treino rotulado entra no ajuste; o teste é apenas pontuado.
        pipeline.fit(X_train, y_train)

        if hasattr(pipeline, "predict_proba"):
            score = pipeline.predict_proba(X_test)[:, 1]
            score_type = "probabilidade da classe 1"
        else:
            score = pipeline.decision_function(X_test)
            score_type = "decision_function (não é probabilidade)"
        decision_threshold = 0.0 if name == "svc_rbf" else 0.5
        predicted_class = (score >= decision_threshold).astype(int)
        score_columns[f"{name}_score"] = score
        score_columns[f"{name}_predicted_class"] = predicted_class
        score_types[name] = score_type

        row: dict[str, Any] = {
            "Modelo": name,
            "Tipo de score": score_type,
            "Limiar padrão": decision_threshold,
        }
        if y_test is not None:
            y_array = y_test.to_numpy()
            row["AUC-ROC"] = (
                float(roc_auc_score(y_array, score)) if np.unique(y_array).size == 2 else None
            )
            tn, fp, fn, tp = confusion_matrix(y_array, predicted_class, labels=[0, 1]).ravel()
            row.update({
                "Sensibilidade": _safe_ratio(int(tp), int(tp + fn)),
                "Especificidade": _safe_ratio(int(tn), int(tn + fp)),
                "F1": float(f1_score(y_array, predicted_class, zero_division=0)),
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            })
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics = metrics.reset_index(drop=True)

    predictions = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], **score_columns})
    prediction_path = root / "results" / "official_test_model_scores.csv"
    predictions.to_csv(prediction_path, index=False)

    summary: dict[str, Any] = {
        "test_file": test_filename,
        "training_rows_used_for_refit": int(len(X_train)),
        "models_scored": MODEL_NAMES,
        "decision_threshold_rule": "estimator default: 0.5 for predict_proba models; 0.0 for SVC decision_function; no threshold tuning",
        "test_rows_scored": int(len(X_test)),
        "test_labels_available": y_test is not None,
        "hyperparameters_source": "stage4_model_search.json; final stratified CV search on all labeled training rows",
        "test_labels_used_for_training_or_tuning": False,
        "test_labels_used_only_for_post_prediction_metrics": y_test is not None,
        "score_type_by_model": score_types,
        "models": json.loads(metrics.to_json(orient="records")),
        "row_level_scores_file": "results/official_test_model_scores.csv",
        "row_level_scores_are_git_ignored": True,
    }
    summary_path = root / "results" / "official_test_metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    figure = None
    if y_test is not None and y_test.nunique() == 2:
        figure, ax = plt.subplots(figsize=(7.5, 6.5))
        for name in MODEL_NAMES:
            score = score_columns[f"{name}_score"]
            fpr, tpr, _ = roc_curve(y_test, score, pos_label=1)
            auc_value = roc_auc_score(y_test, score)
            ax.plot(fpr, tpr, linewidth=1.7, label=f"{name.replace('_', ' ')} (AUC={auc_value:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1.02),
            xlabel="Taxa de falsos positivos",
            ylabel="Sensibilidade",
            title="ROC dos seis modelos no arquivo de teste",
        )
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(root / "results" / "official_test_roc.png", dpi=180, bbox_inches="tight")

    return metrics, predictions, figure, summary
