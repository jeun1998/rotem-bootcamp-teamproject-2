from datetime import date

import pytest

from app import create_app
from app.predictor import DataLimit, DummyPredictor, Prediction, PredictionUnavailable


class FixedPredictor:
    def __init__(self, limit=None):
        self.last_request = None
        self.limit = limit

    def data_limit(self, today, until):
        return self.limit

    def predict_many(self, requests):
        return [self.predict(r) for r in requests]

    def predict(self, request):
        self.last_request = request
        return Prediction(0.25, 0.03, "fixed")


TODAY = date(2024, 6, 1)  # 테스트용 고정 "오늘"
VALID = {"airline": "AA", "origin": "JFK", "dest": "LAX", "date": "2024-07-04", "dep_time": "17:30", "arr_time": "20:45"}


@pytest.fixture
def predictor():
    return FixedPredictor()


@pytest.fixture
def client(predictor):
    return create_app(predictor, today=lambda: TODAY).test_client()


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


def test_predict_builds_service_flight(client, predictor):
    client.post("/api/predict", json={**VALID, "airline": "aa", "dep_time": "07:05"})
    req = predictor.last_request
    assert req.as_of == TODAY
    assert req.to_service_flight() == {
        "Reporting_Airline": "AA",
        "Flight_Number_Reporting_Airline": "",
        "Origin": "JFK",
        "Dest": "LAX",
        "FlightDate": "2024-07-04",
        "CRSDepTime": "0705",
        "CRSArrTime": "2045",
    }


@pytest.mark.parametrize(
    "override, field",
    [
        ({"airline": "ZZ"}, "airline"),
        ({"origin": "XXX"}, "origin"),
        ({"dest": "JFK"}, "dest"),
        ({"date": "2024-13-01"}, "date"),
        ({"date": "2024-05-31"}, "date"),  # 어제
        ({"date": "2025-06-02"}, "date"),  # 366일 뒤
        ({"dep_time": "25:00"}, "dep_time"),
        ({"dep_time": "noon"}, "dep_time"),
        ({"arr_time": ""}, "arr_time"),
        ({"arr_time": "24:00"}, "arr_time"),
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
    app = create_app(DummyPredictor(), today=lambda: TODAY).test_client()
    a = app.post("/api/predict", json=VALID).get_json()
    b = app.post("/api/predict", json=VALID).get_json()
    assert a == b
    assert 0 <= a["delay_probability"] <= 1
    assert 0 <= a["cancel_probability"] <= 1
    assert a["model_version"] == "dummy-0"


@pytest.mark.parametrize("d", ["2024-06-01", "2025-06-01"])  # 오늘, 365일 뒤
def test_predict_accepts_booking_window_edges(client, d):
    assert client.post("/api/predict", json={**VALID, "date": d}).status_code == 200


def test_hourly_varies_only_departure_hour(client, predictor):
    seen = []
    predictor.predict = lambda r: (seen.append(r), Prediction(0.1, 0.01, "fixed"))[1]
    res = client.post("/api/predict/hourly", json=VALID)
    assert res.status_code == 200
    data = res.get_json()
    assert data["model_version"] == "fixed"
    assert [h["hour"] for h in data["hours"]] == list(range(6, 23))
    assert data["hours"][0] == {"hour": 6, "delay_probability": 0.1, "cancel_probability": 0.01}
    assert {(r.airline, r.origin, r.dest, r.flight_date) for r in seen} == {("AA", "JFK", "LAX", date(2024, 7, 4))}
    assert [(r.dep_hour, r.dep_minute) for r in seen] == [(h, 0) for h in range(6, 23)]
    # 예정 소요시간 3시간 15분 유지: 17:30 -> 20:45 이므로 06:00 -> 09:15, 22:00 -> 다음날 01:15
    assert (seen[0].arr_hour, seen[0].arr_minute) == (9, 15)
    assert (seen[-1].arr_hour, seen[-1].arr_minute) == (1, 15)


def test_hourly_validates_like_predict(client):
    res = client.post("/api/predict/hourly", json={**VALID, "origin": "XXX"})
    assert res.status_code == 400
    assert "origin" in res.get_json()["errors"]


def test_sky_video_assets_served(client):
    for path in ("/static/video/sky-loop.mp4", "/static/video/sky-loop-poster.jpg"):
        res = client.get(path)
        assert res.status_code == 200, path
        res.close()


def test_options_date_range_is_booking_window_without_data_limit(client):
    dates = client.get("/api/options").get_json()["dates"]
    assert dates == {"min": "2024-06-01", "max": "2025-06-01", "limit_note": None}


@pytest.fixture
def limited_client():
    limit = DataLimit(last_date=date(2024, 9, 30), data_until="2023-12")
    return create_app(FixedPredictor(limit), today=lambda: TODAY).test_client()


def test_options_date_range_stops_at_data_limit_with_reason(limited_client):
    dates = limited_client.get("/api/options").get_json()["dates"]
    assert dates["max"] == "2024-09-30"
    assert "2023년 12월" in dates["limit_note"] and "2024년 9월 30일" in dates["limit_note"]


def test_date_past_data_limit_is_rejected_with_reason(limited_client):
    assert limited_client.post("/api/predict", json={**VALID, "date": "2024-09-30"}).status_code == 200
    res = limited_client.post("/api/predict", json={**VALID, "date": "2024-10-01"})
    assert res.status_code == 400
    assert "운항 통계가 2023년 12월까지" in res.get_json()["errors"]["date"]


def test_prediction_unavailable_becomes_date_error(client, predictor):
    def unavailable(request):
        raise PredictionUnavailable("기준월 2026-08 이 집계표 범위 밖")
    predictor.predict = unavailable
    res = client.post("/api/predict", json=VALID)
    assert res.status_code == 400
    assert res.get_json() == {"errors": {"date": "이 날짜는 아직 예측할 수 없어요."}}


def test_options_include_the_routes_each_airline_flies(client):
    routes = client.get("/api/options").get_json()["routes"]
    assert "LAX" in routes["AA"]["JFK"]
    assert "JFK" not in routes["WN"]  # 사우스웨스트는 JFK 에 취항하지 않는다


def test_route_the_airline_does_not_fly_is_rejected(client):
    res = client.post("/api/predict", json={**VALID, "airline": "WN"})
    assert res.status_code == 400
    assert res.get_json()["errors"] == {"dest": "이 항공사는 이 노선을 운항하지 않아요."}
