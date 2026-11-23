import pytest

from app import create_app
from app.predictor import DummyPredictor, Prediction


class FixedPredictor:
    def __init__(self):
        self.last_request = None

    def predict(self, request):
        self.last_request = request
        return Prediction(0.25, 0.03, "fixed")


VALID = {"airline": "AA", "origin": "JFK", "dest": "LAX", "date": "2024-07-04", "dep_time": "17:30"}


@pytest.fixture
def predictor():
    return FixedPredictor()


@pytest.fixture
def client(predictor):
    return create_app(predictor).test_client()


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"<form" in res.data


def test_options_lists_airlines_and_airports(client):
    data = client.get("/api/options").get_json()
    assert any(a["code"] == "AA" for a in data["airlines"])
    assert any(a["code"] == "JFK" for a in data["airports"])


def test_predict_returns_probabilities(client):
    res = client.post("/api/predict", json=VALID)
    assert res.status_code == 200
    assert res.get_json() == {"delay_probability": 0.25, "cancel_probability": 0.03, "model_version": "fixed"}


def test_predict_builds_bts_feature_row(client, predictor):
    client.post("/api/predict", json={**VALID, "airline": "aa"})
    row = predictor.last_request.to_feature_row()
    assert row == {
        "FlightDate": "2024-07-04",
        "Month": 7,
        "DayofMonth": 4,
        "DayOfWeek": 4,  # 2024-07-04 은 목요일
        "CRSDepTime": 1730,
        "Reporting_Airline": "AA",
        "Origin": "JFK",
        "Dest": "LAX",
    }


@pytest.mark.parametrize(
    "override, field",
    [
        ({"airline": "ZZ"}, "airline"),
        ({"origin": "XXX"}, "origin"),
        ({"dest": "JFK"}, "dest"),
        ({"date": "2024-13-01"}, "date"),
        ({"dep_time": "25:00"}, "dep_time"),
        ({"dep_time": "noon"}, "dep_time"),
    ],
)
def test_predict_rejects_invalid_input(client, override, field):
    res = client.post("/api/predict", json={**VALID, **override})
    assert res.status_code == 400
    assert field in res.get_json()["errors"]


def test_predict_requires_json(client):
    res = client.post("/api/predict", data="nope")
    assert res.status_code == 400


def test_app_falls_back_to_dummy_without_model_files(tmp_path):
    from app.predictor import load_predictor

    assert isinstance(load_predictor(tmp_path), DummyPredictor)


def test_dummy_is_deterministic_and_bounded():
    app = create_app(DummyPredictor()).test_client()
    a = app.post("/api/predict", json=VALID).get_json()
    b = app.post("/api/predict", json=VALID).get_json()
    assert a == b
    assert 0 <= a["delay_probability"] <= 1
    assert 0 <= a["cancel_probability"] <= 1
    assert a["model_version"] == "dummy-0"
