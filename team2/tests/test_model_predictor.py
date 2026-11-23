"""ADR-0002 계약: models/model.joblib + 데이터 팀 service_features 로 예측한다."""

import math
import sys
from datetime import date
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from app import create_app
from app.predictor import ModelArtifact, ModelPredictor, PredictionRequest, load_predictor

TODAY = date(2024, 6, 1)
BODY = {"airline": "AA", "origin": "ORD", "dest": "LAX", "date": "2024-07-04", "dep_time": "17:30", "arr_time": "20:45"}
FEATURES = ["Origin", "dep_hour_local", "tail_leg_index_hist"]

FAKE_SERVICE_FEATURES = '''
import pandas as pd

CALLS = []
BATCHES = []

class StaleAggregatesError(ValueError):
    pass

def aggregate_range():
    return ("2020-11", "2024-12", None)

def build_service_features(flight, as_of_date, features=None):
    CALLS.append((flight, as_of_date, features))
    if flight["CRSDepTime"] == "0313":
        raise RuntimeError("aggregates 조회 실패")
    if flight["FlightDate"] >= "2025-03-01":
        raise StaleAggregatesError("기준월이 집계표 범위 밖")
    return {"Origin": flight["Origin"], "dep_hour_local": int(flight["CRSDepTime"]) // 100, "source_max_month": "2024-03"}

def build_service_features_batch(flights, as_of_date, features=None):
    BATCHES.append(len(flights))
    return pd.DataFrame([build_service_features(f, as_of_date, features) for f in flights])

def service_feature_names():
    return ["Origin", "dep_hour_local", "tail_leg_index_hist", "Distance"]
'''


def request(origin="ORD"):
    return PredictionRequest("AA", origin, "LAX", date(2024, 7, 4), 17, 30, 20, 45, as_of=TODAY)


class FixedModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self, proba):
        self.proba = proba
        self.seen = None

    def predict_proba(self, X):
        self.seen = X
        return np.array([self.proba])


def artifact(model, **overrides):
    raw = {"model": model, "features": FEATURES, "categories": {"Origin": ["ATL", "JFK", "ORD"]},
           "normal_sample_rate": 1.0, "version": "test-1", **overrides}
    return ModelArtifact.parse(raw)


def build_from_flight(flight, as_of_date, features=None):
    return {"Origin": flight["Origin"], "dep_hour_local": 17}


FEATURES_MODULE = SimpleNamespace(build_service_features_batch=lambda flights, as_of_date, features=None:
                                  pd.DataFrame([build_from_flight(f, as_of_date) for f in flights]),
                                  StaleAggregatesError=type("StaleAggregatesError", (ValueError,), {}),
                                  aggregate_range=lambda: ("2020-11", "2024-12", None))


def test_categorical_uses_training_levels_not_single_row_codes():
    model = FixedModel([0.7, 0.2, 0.1])
    ModelPredictor(artifact(model), FEATURES_MODULE).predict(request("ORD"))
    origin = model.seen["Origin"]
    assert list(origin.cat.categories) == ["ATL", "JFK", "ORD"]
    assert origin.cat.codes.iloc[0] == 2


def test_unseen_category_becomes_missing():
    model = FixedModel([0.7, 0.2, 0.1])
    ModelPredictor(artifact(model), FEATURES_MODULE).predict(request("SFO"))
    assert model.seen["Origin"].cat.codes.iloc[0] == -1


def test_feature_absent_from_builder_is_missing_value():
    model = FixedModel([0.7, 0.2, 0.1])
    ModelPredictor(artifact(model), FEATURES_MODULE).predict(request())
    assert list(model.seen.columns) == FEATURES
    assert math.isnan(model.seen["tail_leg_index_hist"].iloc[0])


def test_normal_class_sampling_is_undone():
    p = ModelPredictor(artifact(FixedModel([0.4, 0.5, 0.1]), normal_sample_rate=0.2), FEATURES_MODULE).predict(request())
    total = 0.4 / 0.2 + 0.5 + 0.1
    assert p.delay_probability == pytest.approx(0.5 / total)
    assert p.cancel_probability == pytest.approx(0.1 / total)


@pytest.mark.parametrize("override, message", [
    ({"normal_sample_rate": None}, "normal_sample_rate"),
    ({"categories": {"route": ["A_B"]}}, "route"),
    ({"model": type("M", (), {"classes_": np.array([0, 1])})()}, "classes_"),
])
def test_artifact_rejects_contract_violations(override, message):
    raw = {"model": FixedModel([1, 0, 0]), "features": FEATURES, "categories": {}, "normal_sample_rate": 1.0,
           "version": "v", **override}
    if override.get("normal_sample_rate", 1) is None:
        del raw["normal_sample_rate"]
    with pytest.raises(ValueError, match=message):
        ModelArtifact.parse(raw)


@pytest.fixture
def deployed(tmp_path):
    """실제 sklearn 모델을 계약대로 저장하고, 가짜 데이터 팀 src 폴더를 만든다."""
    rng = np.random.default_rng(0)
    origin = rng.choice(["ATL", "JFK", "ORD"], 600)
    X = pd.DataFrame({"Origin": pd.Categorical(origin, categories=["ATL", "JFK", "ORD"]),
                      "dep_hour_local": rng.integers(6, 23, 600),
                      "tail_leg_index_hist": np.where(rng.random(600) < 0.5, np.nan, rng.integers(1, 6, 600))})
    y = np.select([origin == "ATL", origin == "JFK"], [0, 1], 2)
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=20).fit(X, y)
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump({"model": model, "features": FEATURES, "categories": {"Origin": ["ATL", "JFK", "ORD"]},
                 "normal_sample_rate": 1.0, "version": "hgb-test"}, models / "model.joblib")
    src = tmp_path / "src"
    src.mkdir()
    (src / "service_features.py").write_text(FAKE_SERVICE_FEATURES, encoding="utf-8")
    yield models, src
    sys.modules.pop("service_features", None)
    if str(src) in sys.path:
        sys.path.remove(str(src))


def client_for(models, src):
    return create_app(load_predictor(models, features_src=str(src)), today=lambda: TODAY).test_client()


def test_deployed_model_predicts_through_service_features(deployed):
    client = client_for(*deployed)
    jfk = client.post("/api/predict", json={**BODY, "origin": "JFK"}).get_json()
    ord_ = client.post("/api/predict", json=BODY).get_json()
    assert jfk["model_version"] == "hgb-test"
    assert jfk["delay_probability"] > 0.5      # 학습 데이터에서 JFK 는 지연
    assert ord_["cancel_probability"] > 0.5    # ORD 는 결항
    import service_features
    flight, as_of, features = service_features.CALLS[-1]
    assert (flight["CRSDepTime"], flight["CRSArrTime"], as_of) == ("1730", "2045", "2024-06-01")
    assert features == FEATURES


def test_hourly_shifts_arrival_for_each_hour(deployed):
    client_for(*deployed).post("/api/predict/hourly", json=BODY)
    import service_features
    first, last = service_features.CALLS[-17][0], service_features.CALLS[-1][0]
    assert (first["CRSDepTime"], first["CRSArrTime"]) == ("0600", "0915")
    assert (last["CRSDepTime"], last["CRSArrTime"]) == ("2200", "0115")


def test_feature_builder_failure_returns_json_error(deployed):
    res = client_for(*deployed).post("/api/predict", json={**BODY, "dep_time": "03:13"})
    assert res.status_code == 500
    assert "_" in res.get_json()["errors"]


def test_missing_features_module_fails_at_load(deployed, tmp_path):
    with pytest.raises(RuntimeError, match="service_features.py"):
        load_predictor(deployed[0], features_src=str(tmp_path / "nowhere"))


def test_model_feature_the_service_cannot_build_fails_at_load(deployed):
    models, src = deployed
    raw = joblib.load(models / "model.joblib")
    joblib.dump({**raw, "features": FEATURES + ["wind_max_kt"]}, models / "model.joblib")
    with pytest.raises(RuntimeError, match="wind_max_kt"):
        load_predictor(models, features_src=str(src))


def test_without_model_file_uses_dummy(tmp_path):
    assert type(load_predictor(tmp_path)).__name__ == "DummyPredictor"


def test_date_range_ends_before_first_month_the_data_cannot_cover(deployed):
    client = client_for(*deployed)
    dates = client.get("/api/options").get_json()["dates"]
    assert dates["max"] == "2025-02-28"
    assert "2024년 12월" in dates["limit_note"]
    res = client.post("/api/predict", json={**BODY, "date": "2025-03-01"})
    assert res.status_code == 400
    assert "2025년 2월 28일 이후" in res.get_json()["errors"]["date"]


def test_hourly_builds_all_departures_in_one_call_and_matches_single(deployed):
    client = client_for(*deployed)
    client.get("/api/options")  # 그날의 예측 가능 기간 확인(달마다 한 편)은 먼저 끝내 둔다
    import service_features
    before = len(service_features.BATCHES)
    hourly = client.post("/api/predict/hourly", json=BODY).get_json()["hours"]
    assert service_features.BATCHES[before:] == [17]
    for h in (6, 17, 22):
        single = client.post("/api/predict", json={**BODY, "dep_time": f"{h:02d}:00"}).get_json()
        row = next(x for x in hourly if x["hour"] == h)
        assert (row["delay_probability"], row["cancel_probability"]) == \
               (single["delay_probability"], single["cancel_probability"])


def test_temperature_scales_probabilities_before_reporting():
    p = ModelPredictor(artifact(FixedModel([0.7, 0.2, 0.1]), temperature=2.0), FEATURES_MODULE).predict(request())
    scaled = np.sqrt([0.7, 0.2, 0.1])  # softmax(log p / 2) = sqrt(p) 정규화
    scaled /= scaled.sum()
    assert p.delay_probability == pytest.approx(scaled[1])
    assert p.cancel_probability == pytest.approx(scaled[2])


def test_artifact_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        artifact(FixedModel([1, 0, 0]), temperature=0)


def test_unreadable_model_file_says_to_match_library_versions(deployed):
    models, src = deployed
    (models / "model.joblib").write_bytes(b"not a pickle")
    with pytest.raises(RuntimeError, match="requirements.txt"):
        load_predictor(models, features_src=str(src))
