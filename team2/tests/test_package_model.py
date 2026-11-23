"""ml/package_model.py: 모델 팀 결과(Pipeline + metadata.json)를 ADR-0002 형식으로 묶는다."""

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from app import create_app
from app.predictor import load_predictor

spec = importlib.util.spec_from_file_location("package_model", Path(__file__).parents[1] / "ml" / "package_model.py")
package_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_model)

TODAY = date(2024, 6, 1)
FEATURES = ["Origin", "dep_hour_local"]
TEMPERATURE = 1.3
FAKE_SERVICE_FEATURES = """
import pandas as pd

class StaleAggregatesError(ValueError):
    pass

def aggregate_range():
    return ("2020-11", "2030-12", None)

def build_service_features_batch(flights, as_of_date, features=None):
    return pd.DataFrame([{"Origin": f["Origin"], "dep_hour_local": int(f["CRSDepTime"]) // 100} for f in flights])

def service_feature_names():
    return ["Origin", "dep_hour_local"]
"""


def team_pipeline():
    """모델 팀 train.py 와 같은 모양: 문자열 범주를 OrdinalEncoder 로, 나머지는 그대로."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"Origin": rng.choice(["ATL", "JFK", "ORD"], 600), "dep_hour_local": rng.integers(6, 23, 600)})
    y = np.where(X["dep_hour_local"] >= 17, 1, np.where(X["Origin"] == "ORD", 2, 0))
    pre = ColumnTransformer([("cats", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["Origin"]),
                             ("nums", "passthrough", ["dep_hour_local"])])
    return Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(max_iter=30))]).fit(X, y)


@pytest.fixture
def team_outputs(tmp_path):
    out = tmp_path / "outputs"
    (out / "models").mkdir(parents=True)
    joblib.dump(team_pipeline(), out / "models" / "no_weather.joblib")
    meta = {"training_correction": "sample_weight only", "data_version": "processed_v3",
            "no_weather": {"features": FEATURES, "categorical": ["Origin"], "temperature": TEMPERATURE}}
    (out / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return out


@pytest.fixture
def features_src(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "service_features.py").write_text(FAKE_SERVICE_FEATURES, encoding="utf-8")
    yield src
    sys.modules.pop("service_features", None)
    if str(src) in sys.path:
        sys.path.remove(str(src))


def test_packaged_model_predicts_like_the_team_pipeline(team_outputs, features_src, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(package_model.package(team_outputs), models / "model.joblib")
    client = create_app(load_predictor(models, features_src=str(features_src)), today=lambda: TODAY).test_client()
    body = {"airline": "AA", "origin": "ORD", "dest": "JFK", "date": "2024-07-04", "dep_time": "19:00", "arr_time": "22:10"}
    served = client.post("/api/predict", json=body).get_json()

    raw = team_pipeline().predict_proba(pd.DataFrame({"Origin": ["ORD"], "dep_hour_local": [19]}))
    logits = np.log(raw) / TEMPERATURE
    expected = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert served["delay_probability"] == pytest.approx(expected[0, 1], abs=1e-4)
    assert served["cancel_probability"] == pytest.approx(expected[0, 2], abs=1e-4)
    assert served["model_version"].startswith("processed_v3-no_weather-")


def test_files_side_by_side_in_one_folder(team_outputs, tmp_path):
    flat = tmp_path / "models"
    flat.mkdir()
    (flat / "metadata.json").write_bytes((team_outputs / "metadata.json").read_bytes())
    (flat / "no_weather.joblib").write_bytes((team_outputs / "models" / "no_weather.joblib").read_bytes())
    assert package_model.package(flat)["temperature"] == TEMPERATURE


def test_lfs_pointer_is_rejected(team_outputs):
    (team_outputs / "models" / "no_weather.joblib").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(SystemExit, match="git lfs pull"):
        package_model.package(team_outputs)


def test_other_training_correction_is_rejected(team_outputs):
    meta = json.loads((team_outputs / "metadata.json").read_text())
    (team_outputs / "metadata.json").write_text(json.dumps({**meta, "training_correction": "none"}))
    with pytest.raises(SystemExit, match="sample_weight"):
        package_model.package(team_outputs)


def test_xgboost_model_is_stored_without_platform_specific_state(team_outputs, tmp_path):
    import io

    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"Origin": rng.choice(["ATL", "JFK", "ORD"], 300), "dep_hour_local": rng.integers(6, 23, 300)})
    y = np.where(X["dep_hour_local"] >= 17, 1, np.where(X["Origin"] == "ORD", 2, 0))
    pre = ColumnTransformer([("cats", OrdinalEncoder(), ["Origin"]), ("nums", "passthrough", ["dep_hour_local"])])
    pipe = Pipeline([("pre", pre), ("model", XGBClassifier(n_estimators=10))]).fit(X, y)
    joblib.dump(pipe, team_outputs / "models" / "no_weather.joblib")

    buf = io.BytesIO()
    joblib.dump(package_model.package(team_outputs), buf)
    # XGBoost 의 기본 피클은 OS 마다 형식이 다른 rng_state 를 담아 Linux 에서 묶은 모델이 Windows 에서 안 열렸다
    assert b"rng_state" not in buf.getvalue()
    buf.seek(0)
    restored = joblib.load(buf)["model"]
    assert list(restored.classes_) == [0, 1, 2]
    assert np.array_equal(restored.predict_proba(X), pipe.predict_proba(X))
