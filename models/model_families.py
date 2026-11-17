"""Shared, regularized classifiers used for audio and image comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight


MODEL_ARTIFACTS = {
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}


def build_models() -> dict[str, Any]:
    """Create the same small, regularized classifier set for each modality."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=42,
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
                cv=3,
            ),
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=9, weights="uniform"),
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=75,
            learning_rate=0.05,
            max_depth=1,
            min_samples_leaf=6,
            subsample=0.7,
            random_state=42,
        ),
    }


def train_comparison_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    modality: str,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    """Train each selected family, report holdout metrics, and save inference artifacts."""
    models = build_models()
    selected_names = list(MODEL_ARTIFACTS) if model_names is None else model_names
    unknown_names = set(selected_names) - set(models)
    if unknown_names:
        raise ValueError(f"Unknown model families: {', '.join(sorted(unknown_names))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"modality": modality, "classes": class_names, "models": {}}
    for name in selected_names:
        model = models[name]
        if isinstance(model, GradientBoostingClassifier):
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        validation_predictions = model.predict(X_validation)
        test_predictions = model.predict(X_test)
        model_metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_predictions)),
            "validation_accuracy": float(accuracy_score(y_validation, validation_predictions)),
            "validation_macro_f1": float(f1_score(y_validation, validation_predictions, average="macro", zero_division=0)),
            "test_accuracy": float(accuracy_score(y_test, test_predictions)),
            "test_macro_f1": float(f1_score(y_test, test_predictions, average="macro", zero_division=0)),
        }
        report["models"][name] = model_metrics
        joblib.dump(
            {"model": model, "classes": [str(label) for label in model.classes_]},
            output_dir / MODEL_ARTIFACTS[name],
        )

    report_path = output_dir.parent / "results" / "classifier_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"\n{modality.title()} model comparison")
    print(f"{'Model':<24}{'Train':>10}{'Validation':>13}{'Test':>10}{'Test macro F1':>16}")
    for name, metrics in report["models"].items():
        print(
            f"{name:<24}"
            f"{metrics['train_accuracy']:>9.2%}"
            f"{metrics['validation_accuracy']:>12.2%}"
            f"{metrics['test_accuracy']:>9.2%}"
            f"{metrics['test_macro_f1']:>15.2%}"
        )
    return report
