
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)
from src.core import OUT, load, prepare, scale_prob

# 기존 학습 결과 불러오기
meta = json.loads(
    (OUT / "metadata.json").read_text(encoding="utf-8")
)

model_name = meta["winner"]
model = joblib.load(
    OUT / "models" / f"{model_name}.joblib"
)

# 2024년 Validation만 사용
data = load("validation", meta["features"])
X = prepare(
    data,
    meta["features"],
    meta["categorical"],
    meta["elapsed_route_medians"],
    meta["elapsed_median"]
)

y = (data["label"].to_numpy() == 2).astype(int)

proba = scale_prob(
    model.predict_proba(X),
    meta["temperature"]
)[:, 2]

precision, recall, thresholds = precision_recall_curve(
    y, proba
)

ap = average_precision_score(y, proba)

# Precision 30%, 50% 달성 시 최대 Recall
for target in [0.30, 0.50]:
    valid = np.where(precision[:-1] >= target)[0]

    if len(valid) == 0:
        print(f"Precision {target:.0%}: 달성 불가")
        continue

    idx = valid[np.argmax(recall[valid])]

    print(
        f"Precision 목표 {target:.0%} | "
        f"실제 Precision {precision[idx]:.4f} | "
        f"Recall {recall[idx]:.4f} | "
        f"Threshold {thresholds[idx]:.4f}"
    )

# 결항 위험 상위 1%의 실제 결항률
k = max(1, int(len(y) * 0.01))
top_idx = np.argsort(proba)[-k:]

print("결항 AP:", round(ap, 5))
print("전체 결항률:", round(y.mean(), 5))
print("상위 1% 실제 결항률:", round(y[top_idx].mean(), 5))

# PR 곡선 저장
folder = OUT / "plots"
folder.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.plot(
    recall,
    precision,
    label=f"{model_name} (AP={ap:.4f})"
)
plt.axhline(
    y.mean(),
    linestyle="--",
    label="Random baseline"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Cancellation Precision-Recall - Validation 2024")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    folder / "cancel_pr_validation.png",
    dpi=200
)
plt.close()

print("Graph saved.")
