"""예측기: 예측 요청 -> 예측 결과.

Flask는 이 모듈의 `Predictor` 인터페이스만 안다. 실제 모델이 준비되면
`models/` 에 joblib 파일을 두기만 하면 `load_predictor()` 가 자동으로 교체한다.
모델 파일 계약은 docs/adr/0001-model-artifact-includes-preprocessing.md 참고.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

# ADR-0001 의 계약. 모델 Pipeline 이 받는 BTS 원본 컬럼.
FEATURE_COLUMNS = [
    "FlightDate",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "CRSDepTime",
    "Reporting_Airline",
    "Origin",
    "Dest",
]


@dataclass(frozen=True)
class PredictionRequest:
    airline: str
    origin: str
    dest: str
    flight_date: date
    dep_hour: int
    dep_minute: int

    def to_feature_row(self) -> dict:
        d = self.flight_date
        return {
            "FlightDate": d.isoformat(),
            "Month": d.month,
            "DayofMonth": d.day,
            "DayOfWeek": d.isoweekday(),  # BTS: 1=월 ... 7=일
            "CRSDepTime": self.dep_hour * 100 + self.dep_minute,
            "Reporting_Airline": self.airline,
            "Origin": self.origin,
            "Dest": self.dest,
        }


@dataclass(frozen=True)
class Prediction:
    delay_probability: float
    cancel_probability: float
    model_version: str

    def to_dict(self) -> dict:
        return {
            "delay_probability": round(self.delay_probability, 4),
            "cancel_probability": round(self.cancel_probability, 4),
            "model_version": self.model_version,
        }


class Predictor(Protocol):
    def predict(self, request: PredictionRequest) -> Prediction: ...


class DummyPredictor:
    """개발용 더미. 그럴듯하지만 학습된 것이 아닌 결정적(deterministic) 규칙.

    - 저녁 출발일수록 지연 확률 증가 (앞선 편 지연이 누적되는 경향)
    - 여름(6~8월), 연말(12월)에 지연 증가
    - 겨울(12~2월)에 결항 증가
    """

    version = "dummy-0"

    def predict(self, request: PredictionRequest) -> Prediction:
        hour = request.dep_hour
        month = request.flight_date.month

        delay = 0.12 + 0.012 * max(0, hour - 6)
        if month in (6, 7, 8, 12):
            delay += 0.06

        cancel = 0.012
        if month in (12, 1, 2):
            cancel += 0.018

        return Prediction(
            delay_probability=min(delay, 0.95),
            cancel_probability=min(cancel, 0.5),
            model_version=self.version,
        )


class JoblibPredictor:
    """ADR-0001 계약을 따르는 sklearn Pipeline 두 개를 감싼다."""

    def __init__(self, delay_path: Path, cancel_path: Path):
        import joblib  # 실제 모델이 있을 때만 필요

        self._delay = joblib.load(delay_path)
        self._cancel = joblib.load(cancel_path)
        self.version = f"joblib:{delay_path.stem}+{cancel_path.stem}"

    def predict(self, request: PredictionRequest) -> Prediction:
        import pandas as pd

        X = pd.DataFrame([request.to_feature_row()], columns=FEATURE_COLUMNS)
        return Prediction(
            delay_probability=float(self._delay.predict_proba(X)[0, 1]),
            cancel_probability=float(self._cancel.predict_proba(X)[0, 1]),
            model_version=self.version,
        )


def load_predictor(models_dir: Path) -> Predictor:
    delay_path = models_dir / "delay.joblib"
    cancel_path = models_dir / "cancel.joblib"
    if delay_path.exists() and cancel_path.exists():
        return JoblibPredictor(delay_path, cancel_path)
    return DummyPredictor()
