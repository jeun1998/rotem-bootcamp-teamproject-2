"""드롭다운 목록(app/data/airlines.json, airports.json)은 ml/build_options.py 가 집계표에서 만든다."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("build_options", ROOT / "ml" / "build_options.py")
build_options = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_options)


def load(name):
    return json.loads((ROOT / "app" / "data" / f"{name}.json").read_text(encoding="utf-8"))


def test_option_files_match_what_the_aggregates_produce():
    airlines, airports, routes = build_options.build()
    assert load("airlines") == airlines, "python ml/build_options.py 로 다시 만드세요"
    assert load("airports") == airports, "python ml/build_options.py 로 다시 만드세요"
    assert load("routes") == routes, "python ml/build_options.py 로 다시 만드세요"


def test_every_airport_has_a_timezone():
    stations = set(pd.read_csv(ROOT / "ml" / "weather_download" / "station_map.csv")["bts_iata"])
    assert {a["code"] for a in load("airports")} <= stations


def test_codes_and_display_names_are_unique():
    for rows in (load("airlines"), load("airports")):
        assert len({r["code"] for r in rows}) == len(rows)
        shown = [r.get("name_ko", r["name"]) for r in rows]
        assert len(set(shown)) == len(shown)


def test_routes_only_use_listed_airlines_and_airports():
    codes = {a["code"] for a in load("airports")}
    routes = load("routes")
    assert set(routes) <= {a["code"] for a in load("airlines")}
    assert all(o in codes and set(ds) <= codes and o not in ds for by in routes.values() for o, ds in by.items())
