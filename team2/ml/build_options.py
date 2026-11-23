"""드롭다운 목록(app/data/airlines.json, airports.json, routes.json)을 BTS 집계표에서 만든다.

최근 12개월(집계표 마지막 달 기준)에 운항 기록이 있는 항공사와 공항을 운항이 많은 순으로 담는다.
routes.json 은 항공사별로 실제 운항하는 출발→도착 노선이다 (아래 carrier_routes 참고).
이름은 app/data/names.json(사람이 관리)에서 붙이고, 거기 없는 공항은 station_map.csv 이름을 다듬어 영문으로만 쓴다.
집계표를 갱신하면 이 스크립트도 다시 돌린다:
    python ml/build_options.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
AGG = REPO / "data" / "aggregates"
DATA = REPO / "app" / "data"
STATION_MAP = REPO / "ml" / "weather_download" / "station_map.csv"

# station_map 이름 끝의 공항 종류·관측소 표기는 화면에 필요 없다
NOISE = re.compile(r"\s*(\((ASOS|AW|CIV|AWOS)\)|INTL\.?|INTERNATIONAL|ARPT|AIRPORT|APT|MUNI(CIPAL)?|RGNL|REGIONAL|FIELD|"
                   r"AFB|\(AW\)|\(CIV\))\s*", re.I)


def tidy(station_name: str) -> str:
    name = NOISE.sub(" ", station_name.replace("_", " ")).strip(" -/")
    name = re.sub(r"\s+", " ", name)
    return name.title() if name.isupper() or name.islower() else name


MIN_LEGS_PER_YEAR = 12  # 한 달 한 번 꼴. 그보다 드문 구간은 결항·회항으로 끊긴 기체 일정이 만든 가짜 노선이 섞인다
FALLBACK_MIN_PER_YEAR = 52  # 주 1편 이상인데 기체 일정에서 운항사를 못 찾은 노선만 공항 기준으로 채운다


def recent_months(df: pd.DataFrame) -> pd.DataFrame:
    last = pd.Period(df["ym"].max(), freq="M")
    return df[df["ym"] > str(last - 12)]


def recent_counts(table: str, key: str | list[str]) -> pd.Series:
    cols = ["ym", "n"] + ([key] if isinstance(key, str) else key)
    return recent_months(pd.read_parquet(AGG / f"monthly_outcomes_{table}.parquet", columns=cols)).groupby(key)["n"].sum()


def carrier_routes() -> set[tuple[str, str, str]]:
    """(항공사, 출발, 도착). 집계표에 항공사×노선 표가 없어서 기체 일정에서 만든다.

    tail_sameday 의 한 행은 항공사 A 의 기체가 직전에 tail_prev_origin 에서 출발해 Origin 에 왔다는 뜻이므로
    (A, tail_prev_origin, Origin) 은 A 가 운항한 구간이다. 실제 노선 표에 있고 1년에 MIN_LEGS_PER_YEAR 번 이상
    나온 구간만 쓴다. 노선 표에서도 1년에 MIN_LEGS_PER_YEAR 편 이상인 노선만 실제 노선으로 본다. 기체의 하루 마지막 편은 여기 잡히지 않으므로, 반대 방향을 운항하는 항공사가 돌아오는 편도
    운항한다고 보고, 그래도 빠진 주 1편 이상 노선은 두 공항 모두에서 출발하는 항공사에 준다.
    """
    routes = recent_counts("hist_route", ["Origin", "Dest"])
    real = set(routes[routes >= MIN_LEGS_PER_YEAR].index)  # 1년에 몇 편뿐인 노선은 회항편 등이라 노선으로 보지 않는다
    last = pd.Period(pd.read_parquet(AGG / "monthly_outcomes_hist_route.parquet", columns=["ym"])["ym"].max(), freq="M")
    parts = [AGG / "tail_sameday" / f"year={p.year}" / f"month={p.month:02d}" / "part.parquet"
             for p in pd.period_range(last - 11, last, freq="M")]
    t = pd.concat([pd.read_parquet(f, columns=["Reporting_Airline", "tail_prev_origin", "Origin"]) for f in parts if f.exists()])
    legs = t.dropna().groupby(["Reporting_Airline", "tail_prev_origin", "Origin"]).size()
    is_real = np.array([(o, d) in real for _, o, d in legs.index])
    found = set(legs[(legs.to_numpy() >= MIN_LEGS_PER_YEAR) & is_real].index)

    covered = {(o, d) for _, o, d in found}
    found |= {(a, d, o) for a, o, d in found if (d, o) in real and (d, o) not in covered}
    covered = {(o, d) for _, o, d in found}
    departs = recent_counts("hist_carrier_origin", ["Reporting_Airline", "Origin"]).index
    at = {}
    for a, o in departs:
        at.setdefault(o, set()).add(a)
    weekly = set(routes[routes >= FALLBACK_MIN_PER_YEAR].index)
    for o, d in weekly - covered:
        found |= {(a, o, d) for a in at.get(o, set()) & at.get(d, set())}
    return found


def build() -> tuple[list[dict], list[dict], dict]:
    names = json.loads((DATA / "names.json").read_text(encoding="utf-8"))
    stations = pd.read_csv(STATION_MAP).set_index("bts_iata")["station_name"]

    carriers = recent_counts("hist_carrier", "Reporting_Airline").sort_values(ascending=False)
    missing = [c for c in carriers.index if c not in names["airlines"]]
    if missing:
        raise SystemExit(f"app/data/names.json 에 항공사 이름을 추가하세요: {missing}")
    airlines = [{"code": c, **names["airlines"][c]} for c in carriers.index]

    traffic = recent_counts("hist_origin", "Origin").add(recent_counts("hist_dest", "Dest"), fill_value=0)
    # 예정 소요시간을 계산하려면 공항 시간대가 필요하다
    traffic = traffic[traffic.index.isin(stations.index)].sort_values(ascending=False)
    airports = [{"code": c, **names["airports"].get(c, {"name": tidy(stations[c])})} for c in traffic.index]

    rank = {a["code"]: i for i, a in enumerate(airports)}
    routes: dict[str, dict[str, list[str]]] = {}
    for a, o, d in carrier_routes():
        if a in names["airlines"] and o in rank and d in rank:
            routes.setdefault(a, {}).setdefault(o, []).append(d)
    # 목록과 같은 순서(운항이 많은 공항 먼저)로 저장해 결과 파일이 매번 같게 한다
    routes = {a: {o: sorted(ds, key=rank.get) for o, ds in sorted(by.items(), key=lambda kv: rank[kv[0]])}
              for a, by in sorted(routes.items())}
    return airlines, airports, routes


def main() -> None:
    airlines, airports, routes = build()
    for name, rows in (("airlines", airlines), ("airports", airports)):
        (DATA / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "routes.json").write_text(json.dumps(routes, separators=(",", ":")) + "\n", encoding="utf-8")
    named = sum("name_ko" in a for a in airports)
    n_routes = sum(len(ds) for by in routes.values() for ds in by.values())
    print(f"항공사 {len(airlines)}곳, 공항 {len(airports)}곳 (한글 이름 {named}곳), 항공사별 노선 {n_routes}개 → app/data/")


if __name__ == "__main__":
    main()
