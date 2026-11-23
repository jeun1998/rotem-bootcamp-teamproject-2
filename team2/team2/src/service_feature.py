"""웹 서비스용: 한 편의 '날씨를 뺀' 학습 피처를 aggregates/ 만 조회해서 만든다.

규칙 (학습셋과 동일)
  - 운항월 M 의 BTS 기반 피처는 src = min(M-3, as_of 월-3) 월 말까지 공개된 BTS만 사용
  - hist_*        : src-11 ~ src 월 실적 (monthly_outcomes_*), 베이지안 평활 m=50
  - *_proxy_ly    : 운항일 D-364/371/357 의 예정 편수 평균 (sched_counts_*)
  - *_proxy_recent: src 월 중 운항일과 같은 요일인 날들의 평균 편수
  - origin_yoy_ratio: 출발 공항 src 월 편수 / src-12 월 편수
  - tail_*_hist   : 참조일(D-364/371/357 + src 월 같은 요일)의 같은 항공사·편명·출발 공항 편 기체 일정 (tail_sameday/)
  - Distance      : route_distance (src 이하 최신 월), CRSElapsedTime: 예정 출발·도착 시각과 공항 시간대로 계산

사용:
    from service_features import build_service_features
    build_service_features(dict(Reporting_Airline="DL", Flight_Number_Reporting_Airline="1209", Origin="PDX",
                                Dest="SLC", FlightDate="2025-08-23", CRSDepTime="1315", CRSArrTime="1607"),
                           as_of_date="2025-08-20")
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import column_selection as S  # noqa: E402
import config as C  # noqa: E402
import feature_engineering as FE  # noqa: E402
from preprocess_flights import load_station_map, local_naive, localize  # noqa: E402

FMT = "%Y-%m-%d"


@lru_cache(maxsize=None)
def _station_map():
    return load_station_map()


@lru_cache(maxsize=64)
def _hist_table(name, src):
    """src(월 문자열)를 마지막 달로 하는 12개월 창의 hist 표."""
    sp = pd.Period(src, freq="M")
    months = [str(sp - i) for i in range(C.HIST_WINDOW_MONTHS)]
    counts = pd.read_parquet(C.AGG / f"monthly_outcomes_{name}.parquet", filters=[("ym", "in", months)])
    target = str(sp + C.BTS_LAG_MONTHS)
    t = FE.rolling_hist_table(counts, name, [target], C.HIST_WINDOW_MONTHS, C.HIST_SMOOTH_M, lag=C.BTS_LAG_MONTHS)
    return t.drop(columns=["hist_month", "window_first_month", "window_last_month"])


@lru_cache(maxsize=256)
def _sched(name, lo, hi):
    return pd.read_parquet(C.AGG / f"sched_counts_{name}.parquet",
                           filters=[("FlightDate", ">=", lo), ("FlightDate", "<=", hi)])


@lru_cache(maxsize=32)
def _tail_month(period):
    p = pd.Period(period, freq="M")
    f = C.AGG / "tail_sameday" / f"year={p.year}" / f"month={p.month:02d}" / "part.parquet"
    if not f.exists():
        return None
    t = pd.read_parquet(f, columns=["FlightDate"] + FE.TAIL_KEY + ["tail_leg_index_of_day", "tail_sched_turn_min",
                                                                  "tail_prev_origin", "tail_prev_sched_elapsed"])
    t["FlightDate"] = pd.to_datetime(t["FlightDate"])
    return t


@lru_cache(maxsize=1)
def _route_distance():
    return pd.read_parquet(C.AGG / "route_distance.parquet")


def _elapsed_minutes(f, sm):
    """예정 소요시간: 도착 현지 날짜를 출발일 0/+1/-1일 중 0 < (도착 UTC - 출발 UTC) <= 24h 인 가장 이른 값으로."""
    dep, _ = localize(local_naive(f["FlightDate"], f["CRSDepTime"]), f["Origin"].map(sm["timezone"]))
    best = None
    for k in (-1, 0, 1):
        arr, _ = localize(local_naive(f["FlightDate"], f["CRSArrTime"]) + pd.Timedelta(days=k),
                          f["Dest"].map(sm["timezone"]))
        m = (arr - dep).dt.total_seconds().iloc[0] / 60
        if 0 < m <= 24 * 60 and (best is None or m < best):
            best = m
    return best


def build_service_features(flight, as_of_date):
    """flight: Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest, FlightDate(YYYY-MM-DD),
    CRSDepTime(hhmm), CRSArrTime(hhmm). as_of_date: 예측하는 날(이 날 기준 공개된 BTS만 사용).
    반환: 날씨를 뺀 학습 피처 dict + source_max_month."""
    sm = _station_map()
    d = pd.Timestamp(flight["FlightDate"])
    mp = d.to_period("M")
    src = min(C.source_month(mp), pd.Timestamp(as_of_date).to_period("M") - C.BTS_LAG_MONTHS)
    src_end = src.end_time.normalize()

    f = pd.DataFrame([{k: str(flight[k]) for k in ("Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin",
                                                   "Dest", "FlightDate", "CRSDepTime", "CRSArrTime")}])
    f["DayOfWeek"] = d.isoweekday()
    f["CRSElapsedTime"] = _elapsed_minutes(f, sm)
    f["dep_hour_local"] = FE.hhmm_to_minutes(f["CRSDepTime"]) % 1440 // 60
    f["arr_hour_local"] = FE.hhmm_to_minutes(f["CRSArrTime"]) % 1440 // 60
    out = FE.flight_calendar_features(f).iloc[0].to_dict()
    out.update(Reporting_Airline=f.at[0, "Reporting_Airline"], Origin=f.at[0, "Origin"], Dest=f.at[0, "Dest"],
               CRSElapsedTime=f.at[0, "CRSElapsedTime"])
    rd = _route_distance()
    rd = rd[(rd["Origin"] == out["Origin"]) & (rd["Dest"] == out["Dest"]) & (rd["source_month"] <= str(src))]
    out["Distance"] = float(rd.sort_values("source_month")["Distance"].iloc[-1]) if len(rd) else np.nan

    for name, keys in FE.HIST_SPECS.items():
        t = _hist_table(name, str(src))
        row = t.merge(f.assign(dep_hour_local=f["dep_hour_local"].astype(float))[keys], on=keys, how="inner")
        for c in (f"{name}_delay_rate", f"{name}_cancel_rate", f"{name}_n_flights"):
            out[c] = float(row[c].iloc[0]) if len(row) else np.nan

    lags = C.PROXY_LY_LAGS_DAYS
    lo, hi = (d - pd.Timedelta(days=max(lags))).strftime(FMT), (d - pd.Timedelta(days=min(lags))).strftime(FMT)
    if pd.Timestamp(hi) <= src_end:
        ly = FE.congestion_lag_features(f, {k: _sched(k, lo, hi) for k in FE.CONGESTION_SPECS}, lags, "proxy_ly")
        out.update(ly.iloc[0].to_dict())
    else:
        out.update({f"{n}_proxy_ly": np.nan for n in S.CONGESTION_NAMES})
    src_counts = {k: _sched(k, src.start_time.strftime(FMT), src_end.strftime(FMT)) for k in FE.CONGESTION_SPECS}
    out.update(FE.congestion_recent_features(f, src_counts, src).iloc[0].to_dict())
    prev = _sched("origin_sched_deps_day", (src - 12).start_time.strftime(FMT), (src - 12).end_time.strftime(FMT))
    yoy = FE.origin_yoy_table(pd.concat([prev, src_counts["origin_sched_deps_day"]]), src)
    out["origin_yoy_ratio"] = float(yoy.get(out["Origin"], np.nan))

    periods = {src} | {pd.Period(x, freq="M") for x in pd.date_range(d - pd.Timedelta(days=max(C.TAIL_LY_LAGS_DAYS)),
                                                                    d - pd.Timedelta(days=min(C.TAIL_LY_LAGS_DAYS)))}
    refs = [t for t in (_tail_month(str(p)) for p in sorted(periods) if p <= src) if t is not None]
    if refs:
        th = FE.tail_hist_features(f, pd.concat(refs, ignore_index=True), src, C.TAIL_LY_LAGS_DAYS)
        out.update({k: th.iloc[0][k] for k in FE.TAIL_HIST_FEATURES})
    out["source_max_month"] = str(src)
    return out


def service_feature_names():
    """build_service_features 가 만드는 학습 피처 (모든 목록에서 기상 피처를 뺀 것)."""
    lists = S.feature_lists("proxy_ly")
    allf = list(dict.fromkeys(f for k, v in lists.items() for f in v))
    weather = set(S.ORIGIN_WEATHER_FEATURES) | set(S.DEST_WEATHER_FEATURES)
    return [f for f in allf if f not in weather]