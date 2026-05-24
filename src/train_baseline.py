from __future__ import annotations

import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.build_manifest import build_manifest
from src.features import build_feature_table
from src.nppad_paths import FIGURE_ROOT, MODEL_ROOT, REPORT_ROOT


RANDOM_STATE = 42


def save_confusion_matrix(y_true, y_pred, labels) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("NPPAD Accident Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "baseline_confusion_matrix.png", dpi=180)
    plt.close()


def save_feature_importance(model: Pipeline, feature_names: list[str], top_k: int = 25) -> pd.DataFrame:
    rf = model.named_steps["rf"]
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    top = importance.head(top_k).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"], color="#287c71")
    plt.xlabel("Importance")
    plt.title(f"Top {top_k} Random Forest Features")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "baseline_feature_importance.png", dpi=180)
    plt.close()
    return importance


def main() -> None:
    manifest = build_manifest()
    feature_cache = REPORT_ROOT / "features.csv"
    if feature_cache.exists():
        features = pd.read_csv(feature_cache)
    else:
        features = build_feature_table(manifest)
        features.to_csv(feature_cache, index=False, encoding="utf-8-sig")

    train_df = features[features["use_for_training"]].copy()
    meta_cols = {"path", "accident", "severity", "use_for_training"}
    feature_cols = [col for col in train_df.columns if col not in meta_cols]
    X = train_df[feature_cols]
    y = train_df["accident"]

    X_train, X_test, y_train, y_test, severity_train, severity_test = train_test_split(
        X,
        y,
        train_df["severity"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    labels = sorted(y.unique())
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "classes": labels,
    }

    save_confusion_matrix(y_test, y_pred, labels)
    importance = save_feature_importance(classifier, feature_cols)

    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    regressor.fit(X_train, severity_train)
    severity_pred = regressor.predict(X_test)
    metrics["severity_mae"] = mean_absolute_error(severity_test, severity_pred)
    metrics["severity_r2"] = r2_score(severity_test, severity_pred)

    with open(MODEL_ROOT / "baseline_classifier.pkl", "wb") as f:
        pickle.dump({"model": classifier, "feature_cols": feature_cols, "labels": labels}, f)
    with open(MODEL_ROOT / "severity_regressor.pkl", "wb") as f:
        pickle.dump({"model": regressor, "feature_cols": feature_cols}, f)

    pd.DataFrame(report).transpose().to_csv(REPORT_ROOT / "classification_report.csv", encoding="utf-8-sig")
    importance.to_csv(REPORT_ROOT / "feature_importance.csv", index=False, encoding="utf-8-sig")
    (REPORT_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Figures: {FIGURE_ROOT}")
    print(f"Models: {MODEL_ROOT}")


if __name__ == "__main__":
    main()
