"""예측기: 예측 요청 -> 예측 결과.

Flask는 이 모듈의 `Predictor` 인터페이스만 안다. `models/model.joblib` 이 있으면 데이터 팀의
`build_service_features` 로 피처를 만들어 모델에 넣고, 없으면 더미 예측기를 쓴다.
모델 파일 계약은 docs/adr/0002-service-features-and-model-artifact.md 참고.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

MINUTES_PER_DAY = 24 * 60
FEATURES_SRC_ENV = "FLIGHT_FEATURES_SRC"
DEFAULT_FEATURES_SRC = Path(__file__).resolve().parent.parent / "ml" / "data_pipeline"
CLASSES = [0, 1, 2]  # 정상, 지연, 결항 (데이터셋 v3 label)


@dataclass(frozen=True)
class PredictionRequest:
    airline: str
    origin: str
    dest: str
    flight_date: date
    dep_hour: int
    dep_minute: int
    arr_hour: int
    arr_minute: int
    as_of: date  # 예측하는 날. 이 날 기준으로 공개된 BTS만 쓴다

    def with_departure(self, hour: int, minute: int = 0) -> PredictionRequest:
        """출발 시각만 바꾸고 예정 소요시간은 유지한다 (도착 시각을 같은 만큼 옮긴다)."""
        shift = (hour * 60 + minute) - (self.dep_hour * 60 + self.dep_minute)
        arr = (self.arr_hour * 60 + self.arr_minute + shift) % MINUTES_PER_DAY
        return replace(self, dep_hour=hour, dep_minute=minute, arr_hour=arr // 60, arr_minute=arr % 60)

    def to_service_flight(self) -> dict:
        """데이터 팀 `build_service_features` 의 입력 형식 (BTS 컬럼 이름, 현지 hhmm)."""
        return {
            "Reporting_Airline": self.airline,
            "Flight_Number_Reporting_Airline": "",  # 기체 일정 피처에만 쓰인다. 지금은 받지 않는다
            "Origin": self.origin,
            "Dest": self.dest,
            "FlightDate": self.flight_date.isoformat(),
            "CRSDepTime": f"{self.dep_hour:02d}{self.dep_minute:02d}",
            "CRSArrTime": f"{self.arr_hour:02d}{self.arr_minute:02d}",
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


@dataclass(frozen=True)
class DataLimit:
    """피처를 만드는 BTS 집계표가 오래되어, last_date 이후 출발편은 아직 예측할 수 없다."""

    last_date: date
    data_until: str  # 집계표의 마지막 달, "YYYY-MM"

    def _until_ko(self) -> str:
        y, m = self.data_until.split("-")
        return f"{y}년 {int(m)}월"

    def note(self) -> str:
        d = self.last_date
        return f"운항 통계가 {self._until_ko()}까지 반영돼 있어 {d.year}년 {d.month}월 {d.day}일 출발편까지 예측할 수 있어요."

    def error(self) -> str:
        d = self.last_date
        return (f"{d.year}년 {d.month}월 {d.day}일 이후 출발편은 아직 예측할 수 없어요. "
                f"예측에 쓰는 운항 통계가 {self._until_ko()}까지예요.")


class PredictionUnavailable(ValueError):
    """지금 가진 데이터로는 이 요청을 계산할 수 없다. 사용자에게는 날짜 오류로 보인다."""


class Predictor(Protocol):
    def predict(self, request: PredictionRequest) -> Prediction: ...

    def predict_many(self, requests: list[PredictionRequest]) -> list[Prediction]:
        """같은 예측일(as_of)의 여러 요청을 입력 순서대로 예측한다."""

    def data_limit(self, today: date, until: date) -> DataLimit | None:
        """today 부터 until 사이에 예측할 수 없는 출발일이 있으면 그 경계. 없으면 None."""


class DummyPredictor:
    """개발용 더미. 그럴듯하지만 학습된 것이 아닌 결정적(deterministic) 규칙.

    - 저녁 출발일수록 지연 확률 증가 (앞선 편 지연이 누적되는 경향)
    - 여름(6~8월), 연말(12월)에 지연 증가
    - 겨울(12~2월)에 결항 증가
    """

    version = "dummy-0"

    def data_limit(self, today: date, until: date) -> DataLimit | None:
        return None

    def predict_many(self, requests: list[PredictionRequest]) -> list[Prediction]:
        return [self.predict(r) for r in requests]

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


@dataclass(frozen=True)
class ModelArtifact:
    """`models/model.joblib` 에 담긴 dict 를 검증한 것 (ADR-0002)."""

    model: Any
    features: list[str]
    categories: dict[str, list[str]]
    normal_sample_rate: float
    version: str
    temperature: float = 1.0

    REQUIRED = ("model", "features", "categories", "normal_sample_rate", "version")

    @classmethod
    def parse(cls, raw: Any) -> ModelArtifact:
        if not isinstance(raw, dict):
            raise ValueError("model.joblib 은 dict 여야 합니다 (ADR-0002).")
        missing = [k for k in cls.REQUIRED if k not in raw]
        if missing:
            raise ValueError(f"model.joblib 에 필요한 키가 없습니다: {', '.join(missing)} (ADR-0002).")
        classes = getattr(raw["model"], "classes_", None)
        if classes is not None and [int(c) for c in classes] != CLASSES:
            raise ValueError(f"모델 classes_ 는 {CLASSES} (정상, 지연, 결항) 이어야 합니다: {list(classes)}")
        rate = float(raw["normal_sample_rate"])
        if not 0 < rate <= 1:
            raise ValueError(f"normal_sample_rate 는 (0, 1] 이어야 합니다: {rate}")
        unknown_cats = set(raw["categories"]) - set(raw["features"])
        if unknown_cats:
            raise ValueError(f"categories 에 features 에 없는 컬럼이 있습니다: {sorted(unknown_cats)}")
        temperature = float(raw.get("temperature", 1.0))
        if not temperature > 0:
            raise ValueError(f"temperature 는 0보다 커야 합니다: {temperature}")
        return cls(raw["model"], list(raw["features"]), {k: list(v) for k, v in raw["categories"].items()},
                   rate, str(raw["version"]), temperature)


# 예측 가능 기간을 알아낼 때 쓰는 아무 항공편. 데이터 팀의 기준월 판단은 노선과 무관하다
PROBE_FLIGHT = {"Reporting_Airline": "AA", "Flight_Number_Reporting_Airline": "", "Origin": "ATL", "Dest": "BOS",
                "CRSDepTime": "0900", "CRSArrTime": "1130"}


class ModelPredictor:
    def __init__(self, artifact: ModelArtifact, features: ModuleType):
        self._a = artifact
        self._features = features
        self.version = artifact.version
        self._limit_by_day: dict[date, DataLimit | None] = {}

    def _build(self, flights: list[dict], as_of: date):
        try:
            return self._features.build_service_features_batch(flights, as_of_date=as_of.isoformat(),
                                                               features=self._a.features)
        except self._features.StaleAggregatesError as e:
            raise PredictionUnavailable(str(e)) from e

    def data_limit(self, today: date, until: date) -> DataLimit | None:
        if today not in self._limit_by_day:
            self._limit_by_day = {today: self._probe_limit(today, until)}
        return self._limit_by_day[today]

    def _probe_limit(self, today: date, until: date) -> DataLimit | None:
        # 기준월 규칙은 달 단위라서, 달마다 한 편씩 계산해 보고 처음 막히는 달의 전날까지를 예측 가능 기간으로 본다.
        # 규칙을 앱에 옮겨 적지 않으므로 데이터 팀 규칙이나 집계표가 바뀌어도 그대로 맞다
        day = today
        while day <= until:
            try:
                self._build([{**PROBE_FLIGHT, "FlightDate": day.isoformat()}], today)
            except PredictionUnavailable:
                return DataLimit(day - timedelta(days=1), str(self._features.aggregate_range()[1]))
            day = (day.replace(day=1) + timedelta(days=32)).replace(day=1)
        return None

    def predict(self, request: PredictionRequest) -> Prediction:
        return self.predict_many([request])[0]

    def predict_many(self, requests: list[PredictionRequest]) -> list[Prediction]:
        import pandas as pd

        feats = self._build([r.to_service_flight() for r in requests], requests[0].as_of)
        # 참조 데이터가 없는 피처(신규 노선, 기체 일정 미발견 등)는 학습셋처럼 결측으로 둔다
        X = feats.reindex(columns=self._a.features).reset_index(drop=True)
        # 한 행을 그냥 category 로 바꾸면 어떤 값이든 코드 0이 된다. 학습 때 범주 목록으로 맞추고, 없는 값은 결측
        for col, levels in self._a.categories.items():
            X[col] = pd.Categorical(X[col].where(X[col].isin(levels)), categories=levels)
        proba = _apply_temperature(self._a.model.predict_proba(X), self._a.temperature)
        # 학습셋은 정상편만 normal_sample_rate 로 뽑았으므로 정상 확률을 되돌린 뒤 다시 정규화한다
        predictions = []
        for normal, delay, cancel in proba:
            total = normal / self._a.normal_sample_rate + delay + cancel
            predictions.append(Prediction(float(delay / total), float(cancel / total), self.version))
        return predictions


def _apply_temperature(proba, temperature: float):
    """검증셋에서 맞춘 온도 보정: softmax(log p / T). T=1 이면 그대로."""
    import numpy as np

    if temperature == 1.0:
        return proba
    logits = np.log(np.clip(np.asarray(proba, dtype=float), 1e-12, 1)) / temperature
    scaled = np.exp(logits - logits.max(axis=1, keepdims=True))
    return scaled / scaled.sum(axis=1, keepdims=True)


def _import_service_features(src_dir: Path) -> ModuleType:
    if not (src_dir / "service_features.py").exists():
        raise RuntimeError(f"{src_dir} 에 service_features.py 가 없습니다. {FEATURES_SRC_ENV} 로 경로를 지정하세요.")
    sys.path.insert(0, str(src_dir))
    return importlib.import_module("service_features")


def load_predictor(models_dir: Path, features_src: str | None = None) -> Predictor:
    model_path = models_dir / "model.joblib"
    if not model_path.exists():
        return DummyPredictor()

    import joblib

    module = _import_service_features(Path(features_src or os.environ.get(FEATURES_SRC_ENV) or DEFAULT_FEATURES_SRC))
    try:
        raw = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"{model_path} 를 열 수 없습니다. 모델을 저장한 라이브러리 버전과 다를 수 있으니 "
                           f"'pip install -r requirements.txt' 로 xgboost·scikit-learn 버전을 맞추세요. ({e})") from e
    artifact = ModelArtifact.parse(raw)
    unknown = set(artifact.features) - set(module.service_feature_names())
    if unknown:
        raise RuntimeError(f"서비스에서 만들 수 없는 피처가 모델에 있습니다: {sorted(unknown)}")
    return ModelPredictor(artifact, module)
