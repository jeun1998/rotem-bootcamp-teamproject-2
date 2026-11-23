
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic": "Logistic Regression"
}


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOTS / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print("[SAVED]", filename)


# 1. 세 모델의 검증 성능 비교
def plot_model_comparison():
    path = OUT / "validation_comparison.csv"
    df = pd.read_csv(path)

    metrics = {
        "cancel_pr_auc_ap": ("Cancellation AP", "cancel_ap.png"),
        "delay_pr_auc_ap": ("Delay AP", "delay_ap.png"),
        "macro_f1": ("Macro F1", "macro_f1.png")
    }

    for column, (title, filename) in metrics.items():
        plt.figure(figsize=(8, 5))

        names = [
            MODEL_NAMES.get(name, name)
            for name in df["model"]
        ]
        values = df[column].to_numpy()

        bars = plt.bar(names, values)

        plt.title("Validation 2024 - " + title)
        plt.ylabel(title)
        plt.ylim(0, max(values) * 1.23)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom"
            )

        save_plot(filename)


# 2. XGBoost 특성 중요도
def plot_feature_importance():
    path = OUT / "models" / "xgboost.joblib"

    if not path.exists():
        print("[SKIP] XGBoost model not found")
        return

    model = joblib.load(path)

    estimator = model.named_steps["model"]
    processor = model.named_steps["pre"]

    importances = estimator.feature_importances_

    try:
        names = processor.get_feature_names_out()
    except Exception:
        print("[SKIP] Cannot determine feature names")
        return

    if len(names) != len(importances):
        print("[SKIP] Feature importance length mismatch")
        return

    importance = pd.DataFrame({
        "feature": names,
        "importance": importances
    })

    importance["feature"] = (
        importance["feature"]
        .str.replace(r"^(cats|nums)__", "", regex=True)
    )

    importance = importance.sort_values(
        "importance", ascending=False
    ).head(15).iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(
        importance["feature"],
        importance["importance"]
    )
    plt.title("XGBoost - Top 15 Feature Importances")
    plt.xlabel("Feature importance")

    save_plot("feature_importance.png")


# 3. Precision-Recall 곡선
def plot_precision_recall(df):
    plt.figure(figsize=(8, 6))

    for label, column, name in [
        (1, "p_delay", "Delay"),
        (2, "p_cancel", "Cancellation")
    ]:
        actual = (df["actual"] == label).astype(int)

        precision, recall, _ = precision_recall_curve(
            actual, df[column]
        )

        ap = average_precision_score(
            actual, df[column]
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.4f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test 2025 - Precision Recall Curves")
    plt.legend()
    plt.grid(alpha=0.25)

    save_plot("precision_recall.png")


# 4. 실제 클래스와 예측 클래스 분포 비교
def plot_class_distribution(df):
    labels = [0, 1, 2]
    actual = (
        df["actual"].value_counts(normalize=True)
        .reindex(labels, fill_value=0)
    )
    predicted = (
        df["predicted"].value_counts(normalize=True)
        .reindex(labels, fill_value=0)
    )

    x = np.arange(3)
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(
        x - width / 2, actual.values,
        width, label="Actual"
    )
    plt.bar(
        x + width / 2, predicted.values,
        width, label="Predicted"
    )

    plt.xticks(x, ["Normal", "Delay", "Cancel"])
    plt.ylabel("Proportion")
    plt.title("Test 2025 - Class Distribution")
    plt.legend()

    save_plot("class_distribution.png")


# 5. 20만 건과 100만 건의 성능 비교
def plot_train_size_comparison():
    old_path = OUT / "validation_comparison_200k.csv"
    new_path = OUT / "validation_comparison.csv"

    if not old_path.exists():
        print("[SKIP] 200k results not found")
        return

    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)

    merged = old[["model", "cancel_pr_auc_ap"]].merge(
        new[["model", "cancel_pr_auc_ap"]],
        on="model",
        suffixes=("_200k", "_1000k")
    )

    x = np.arange(len(merged))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(
        x - width / 2,
        merged["cancel_pr_auc_ap_200k"],
        width,
        label="200k training rows"
    )
    plt.bar(
        x + width / 2,
        merged["cancel_pr_auc_ap_1000k"],
        width,
        label="1M training rows"
    )

    names = [
        MODEL_NAMES.get(m, m)
        for m in merged["model"]
    ]

    plt.xticks(x, names)
    plt.ylabel("Cancellation Average Precision")
    plt.title("Training Size vs Cancellation AP")
    plt.legend()

    save_plot("train_size_comparison.png")


if __name__ == "__main__":
    plot_model_comparison()
    plot_feature_importance()
    plot_train_size_comparison()

    test_path = OUT / "test_predictions.csv"

    if test_path.exists():
        test = pd.read_csv(test_path)
        plot_precision_recall(test)
        plot_class_distribution(test)
    else:
        print(
            "[INFO] Test plots will be available "
            "after scripts/evaluate.py"
        )

    print("[DONE] PPT graphs:", PLOTS)
