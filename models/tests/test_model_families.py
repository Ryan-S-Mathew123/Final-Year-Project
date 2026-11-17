import joblib
import numpy as np

from models.model_families import MODEL_ARTIFACTS, train_comparison_models


def test_shared_families_train_and_save_holdout_metrics(tmp_path):
    rng = np.random.default_rng(42)
    class_names = ["bird_a", "bird_b", "bird_c"]
    centers = np.eye(3, dtype=np.float32)

    def samples(per_class):
        labels = np.repeat(class_names, per_class)
        features = np.vstack([
            centers[index] + rng.normal(0, 0.1, size=(per_class, 3))
            for index in range(len(class_names))
        ])
        return features, labels

    X_train, y_train = samples(20)
    X_validation, y_validation = samples(6)
    X_test, y_test = samples(6)
    output_dir = tmp_path / "comparison_models"

    report = train_comparison_models(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        class_names,
        output_dir,
        "test",
    )

    assert set(report["models"]) == set(MODEL_ARTIFACTS)
    for metrics in report["models"].values():
        assert {"train_accuracy", "validation_accuracy", "test_accuracy", "test_macro_f1"} <= set(metrics)
        assert all(0 <= metrics[key] <= 1 for key in metrics)
    for filename in MODEL_ARTIFACTS.values():
        artifact = joblib.load(output_dir / filename)
        probabilities = artifact["model"].predict_proba(X_test[:1])
        assert probabilities.shape == (1, len(class_names))
        assert artifact["classes"] == class_names
