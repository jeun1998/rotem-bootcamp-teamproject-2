from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, request

from .predictor import PredictionRequest, Predictor, load_predictor

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = APP_DIR / "data"


def _load_options() -> dict:
    with open(DATA_DIR / "airlines.json", encoding="utf-8") as f:
        airlines = json.load(f)
    with open(DATA_DIR / "airports.json", encoding="utf-8") as f:
        airports = json.load(f)
    return {"airlines": airlines, "airports": airports}


def _parse_request(payload: dict, options: dict) -> tuple[PredictionRequest | None, dict]:
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

    flight_date = None
    try:
        flight_date = date.fromisoformat(str(payload.get("date", "")))
    except ValueError:
        errors["date"] = "날짜는 YYYY-MM-DD 형식이어야 합니다."

    hour = minute = None
    try:
        h, m = str(payload.get("dep_time", "")).split(":")
        hour, minute = int(h), int(m)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except ValueError:
        errors["dep_time"] = "출발 시각은 HH:MM 형식이어야 합니다."

    if errors:
        return None, errors
    return PredictionRequest(airline, origin, dest, flight_date, hour, minute), {}


def create_app(predictor: Predictor | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    options = _load_options()
    app.config["PREDICTOR"] = predictor or load_predictor(ROOT_DIR / "models")

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/options")
    def get_options():
        return jsonify(options)

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"errors": {"_": "JSON 본문이 필요합니다."}}), 400
        req, errors = _parse_request(payload, options)
        if errors:
            return jsonify({"errors": errors}), 400
        return jsonify(app.config["PREDICTOR"].predict(req).to_dict())

    return app
