from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Callable
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .predictor import DataLimit, PredictionRequest, PredictionUnavailable, Predictor, load_predictor

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = APP_DIR / "data"

# 예측 가능 기간: 오늘부터 1년 이내 출발편만 받는다 (CONTEXT.md).
BOOKING_WINDOW_DAYS = 365
# 시간대별 비교에서 보여줄 출발 시각 (06시~22시 정각)
COMPARISON_HOURS = range(6, 23)


def _load_options() -> dict:
    with open(DATA_DIR / "airlines.json", encoding="utf-8") as f:
        airlines = json.load(f)
    with open(DATA_DIR / "airports.json", encoding="utf-8") as f:
        airports = json.load(f)
    with open(DATA_DIR / "routes.json", encoding="utf-8") as f:
        routes = json.load(f)  # 항공사 → 출발 → [도착]. 실제 운항하는 노선 (ml/build_options.py)
    return {"airlines": airlines, "airports": airports, "routes": routes}


def _date_range(predictor: Predictor, today: date) -> tuple[date, DataLimit | None]:
    """받을 수 있는 마지막 출발일과, 그 날을 앞당긴 데이터 한계(있으면)."""
    window_end = today + timedelta(days=BOOKING_WINDOW_DAYS)
    limit = predictor.data_limit(today, window_end)
    if limit is None or limit.last_date >= window_end:
        return window_end, None
    return limit.last_date, limit


def _parse_request(payload: dict, options: dict, today: date, last_date: date,
                   limit: DataLimit | None) -> tuple[PredictionRequest | None, dict]:
    errors: dict[str, str] = {}
    airline_codes = {a["code"] for a in options["airlines"]}
    airport_codes = {a["code"] for a in options["airports"]}

    airline = str(payload.get("airline", "")).upper()
    origin = str(payload.get("origin", "")).upper()
    dest = str(payload.get("dest", "")).upper()

    if airline not in airline_codes:
        errors["airline"] = "지원하지 않는 항공사입니다."
    if origin not in airport_codes:
        errors["origin"] = "지원하지 않는 출발 공항입니다."
    if dest not in airport_codes:
        errors["dest"] = "지원하지 않는 도착 공항입니다."
    if "origin" not in errors and origin == dest:
        errors["dest"] = "출발 공항과 도착 공항이 같습니다."
    elif not errors and dest not in options["routes"].get(airline, {}).get(origin, ()):
        errors["dest"] = "이 항공사는 이 노선을 운항하지 않아요."

    flight_date = None
    try:
        flight_date = date.fromisoformat(str(payload.get("date", "")))
    except ValueError:
        errors["date"] = "날짜는 YYYY-MM-DD 형식이어야 합니다."
    if flight_date is not None:
        if flight_date < today:
            errors["date"] = "이미 지난 날짜는 예측할 수 없습니다."
        elif flight_date > last_date:
            errors["date"] = limit.error() if limit else f"출발일은 오늘부터 {BOOKING_WINDOW_DAYS}일 이내여야 합니다."

    dep = _parse_hhmm(payload.get("dep_time"))
    if dep is None:
        errors["dep_time"] = "출발 시각은 HH:MM 형식이어야 합니다."
    arr = _parse_hhmm(payload.get("arr_time"))
    if arr is None:
        errors["arr_time"] = "도착 시각은 HH:MM 형식이어야 합니다."

    if errors:
        return None, errors
    return PredictionRequest(airline, origin, dest, flight_date, *dep, *arr, as_of=today), {}


def _parse_hhmm(value) -> tuple[int, int] | None:
    try:
        h, m = str(value or "").split(":")
        hour, minute = int(h), int(m)
    except ValueError:
        return None
    return (hour, minute) if 0 <= hour <= 23 and 0 <= minute <= 59 else None


def create_app(
    predictor: Predictor | None = None,
    today: Callable[[], date] = date.today,
) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    options = _load_options()
    app.config["PREDICTOR"] = predictor or load_predictor(ROOT_DIR / "models")

    @app.errorhandler(PredictionUnavailable)
    def unavailable(e):
        app.logger.warning("예측할 수 없는 요청: %s", e)
        return jsonify({"errors": {"date": "이 날짜는 아직 예측할 수 없어요."}}), 400

    @app.errorhandler(Exception)
    def unexpected_error(e):
        if isinstance(e, HTTPException):
            return e
        app.logger.exception("예측 처리 중 오류")
        return jsonify({"errors": {"_": "예측 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요."}}), 500

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/options")
    def get_options():
        last_date, limit = _date_range(app.config["PREDICTOR"], today())
        dates = {"min": today().isoformat(), "max": last_date.isoformat(), "limit_note": limit.note() if limit else None}
        return jsonify({**options, "dates": dates})

    def parse_or_400():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return None, (jsonify({"errors": {"_": "JSON 본문이 필요합니다."}}), 400)
        req, errors = _parse_request(payload, options, today(), *_date_range(app.config["PREDICTOR"], today()))
        if errors:
            return None, (jsonify({"errors": errors}), 400)
        return req, None

    @app.post("/api/predict")
    def predict():
        req, error = parse_or_400()
        if error:
            return error
        return jsonify(app.config["PREDICTOR"].predict(req).to_dict())

    @app.post("/api/predict/hourly")
    def predict_hourly():
        """같은 항공사·노선·날짜에서 출발 시각만 바꿔 가며 예측한다 (예정 소요시간은 유지)."""
        req, error = parse_or_400()
        if error:
            return error
        predictions = app.config["PREDICTOR"].predict_many([req.with_departure(h) for h in COMPARISON_HOURS])
        hours = [{"hour": h, **p.to_dict()} for h, p in zip(COMPARISON_HOURS, predictions)]
        return jsonify({
            "model_version": predictions[0].model_version,
            "hours": [{k: v for k, v in h.items() if k != "model_version"} for h in hours],
        })

    return app
