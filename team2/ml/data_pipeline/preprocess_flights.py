"""BTS 항공편 전처리 (v3): BTS 공개 지연 규칙(운항월 M → M-BTS_LAG_MONTHS 월 말까지)을 지켜 BTS 쪽 피처를 다시 계산.

1단계 (BTS_READ_START ~ 데이터셋 끝, 필요한 컬럼만 월 단위로 읽음) → aggregates/
  - monthly_outcomes_* / hist_*: 그룹×월 실적, 운항월 M 의 창 = M-14 ~ M-3 (source_max_month 포함)
  - sched_counts_*: 날짜×그룹 예정 편수 (source_month 포함)
  - tail_sameday/: 날짜별 당일 기체 일정 (tail_*_hist 의 참조 데이터, source_month 포함)
  - route_distance: 노선 거리 (서비스 입력 보완)
2단계 (데이터셋 각 월)
  - BTS 전체 행: label, 혼잡도 oracle/proxy_ly/proxy_recent (상관 통계용) → v2와 같은 규칙으로 샘플링
  - 샘플된 행: hist_*, origin_yoy_ratio, tail_*_hist, 공개 시점 검사용 bts_source_max_date
  - interim/flights/split=*/year=YYYY/month=MM/part.parquet (BTS 쪽 v3 피처만; 나머지는 v2 병합본 재사용)
사용: python ml/data_pipeline/preprocess_flights.py [--skip-aggregates]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import feature_engineering as FE  # noqa: E402

STR_COLS = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Tail_Number", "Origin",
            "OriginCityName", "OriginState", "Dest", "DestCityName", "DestState", "CRSDepTime", "DepTimeBlk",
            "CRSArrTime", "ArrTimeBlk", "DepTime", "WheelsOff", "WheelsOn", "ArrTime", "CancellationCode"]


def load_station_map():
    sm = pd.read_csv(C.STATION_MAP, dtype=str)
    sm["timezone"] = sm["bts_iata"].map(C.TZ_OVERRIDES).fillna(sm["timezone"])
    return sm.set_index("bts_iata")


def localize(naive, tz_series):
    """공항별 시간대로 현지 naive 시각을 UTC naive로. 반환: (utc, status)."""
    utc = pd.Series(pd.NaT, index=naive.index, dtype="datetime64[ns]")
    status = pd.Series("ok", index=naive.index, dtype=object)
    status[naive.isna()] = "invalid_time"
    status[tz_series.isna()] = "no_timezone"
    for tz, idx in naive[naive.notna() & tz_series.notna()].groupby(tz_series).groups.items():
        t = naive.loc[idx]
        strict = t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        bad = strict.isna()
        if bad.any():
            tb = t[bad]
            # 존재하지 않는 시각(서머타임 시작) -> 앞으로 밀기, 모호한 시각(서머타임 종료) -> 서머타임으로 가정
            fixed = tb.dt.tz_localize(tz, ambiguous=np.ones(len(tb), dtype=bool), nonexistent="shift_forward")
            nonexist = tb.dt.tz_localize(tz, ambiguous=np.ones(len(tb), dtype=bool), nonexistent="NaT").isna()
            status.loc[tb.index[nonexist.to_numpy()]] = "nonexistent_shifted"
            status.loc[tb.index[~nonexist.to_numpy()]] = "ambiguous_dst_assumed"
            strict = strict.copy()
            strict[bad] = fixed
        utc.loc[idx] = strict.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    return utc, status


def local_naive(date_str, hhmm):
    minutes = FE.hhmm_to_minutes(hhmm)
    valid = minutes.between(0, 1440) & (pd.to_numeric(hhmm, errors="coerce") % 100 < 60)
    base = pd.to_datetime(date_str)
    return (base + pd.to_timedelta(minutes.where(valid), unit="m")).astype("datetime64[ns]")


def add_labels(df):
    canc = df["Cancelled"] == 1
    dep = df["DepDelay"]
    df["label"] = np.select([canc, dep >= C.DELAY_THRESHOLD_MIN, dep < C.DELAY_THRESHOLD_MIN], [2, 1, 0], -1)
    df["label_status"] = np.select([canc, dep.notna()], ["cancelled", "defined"], "undefined_no_depdelay")
    return df


def add_label_arr(df):
    """도착 기준 레이블 (평가용, 학습 입력 금지): 결항=2, ArrDel15=1 → 1, 0 → 0, 회항 등 미정의 → -1."""
    arr = df["ArrDel15"]
    df["label_arr"] = np.select([df["Cancelled"] == 1, df["Diverted"] == 1, arr == 1, arr == 0],
                                [2, -1, 1, 0], -1).astype("int8")
    return df


def read_month(year, month, usecols=None):
    f = C.BTS_DIR / f"bts_{year}_{month:02d}.csv.gz"
    df = pd.read_csv(f, dtype={c: str for c in STR_COLS}, usecols=usecols, low_memory=False)
    df["dep_hour_local"] = FE.hhmm_to_minutes(df["CRSDepTime"]) % 1440 // 60
    df["arr_hour_local"] = FE.hhmm_to_minutes(df["CRSArrTime"]) % 1440 // 60
    return df


# ---------------------------------------------------------------- 1단계: 집계표


def sample_month(df, split, year, month):
    """월×클래스 층화 무작위 추출. train은 정상만 NEG_RATE, val/test는 모든 클래스를 EVAL_RATE로."""
    seed = C.SAMPLE_SEED + year * 100 + month
    if split == "train":
        neg = df[df["label"] == 0].sample(frac=C.NEG_RATE, random_state=seed)
        rest = df[df["label"] > 0]
        out = pd.concat([neg.assign(sample_rate=C.NEG_RATE), rest.assign(sample_rate=1.0)])
    else:
        out = (df[df["label"] >= 0].groupby("label", group_keys=False)
               .sample(frac=C.EVAL_RATE, random_state=seed).assign(sample_rate=C.EVAL_RATE))
    out["sample_weight"] = 1.0 / out["sample_rate"]
    return out.sort_values("flight_id")


def utc_departure(df, sm):
    """예정 출발 UTC (출발 공항 시간대)."""
    utc, _ = localize(local_naive(df["FlightDate"], df["CRSDepTime"]), df["Origin"].map(sm["timezone"]))
    return utc


# ---------------------------------------------------------------- 1단계: 집계표 (BTS 전체, 샘플 전)
AGG_COLS = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Tail_Number", "Origin", "Dest",
            "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "Cancelled", "DepDelay"]
TAIL_REF_COLS = ["FlightDate"] + FE.TAIL_KEY + ["tail_leg_index_of_day", "tail_sched_turn_min", "tail_prev_origin",
                                               "tail_prev_sched_elapsed"]


def build_aggregates(sm):
    """과거 실적·예정 편수·당일 기체 일정(참조용)·노선 거리. 모든 표에 source(_max)_month 를 단다."""
    read_months = pd.period_range(C.BTS_READ_START, max(b for _, b in C.SPLITS.values()), freq="M")
    outcome = {k: [] for k in FE.HIST_SPECS}
    sched = {k: [] for k in FE.CONGESTION_SPECS}
    dist = []
    n_rows = 0
    for p in read_months:
        df = read_month(p.year, p.month, AGG_COLS)
        n_rows += len(df)
        for k, v in FE.monthly_outcome_counts(df).items():
            outcome[k].append(v)
        for k, v in FE.daily_schedule_counts(df).items():
            sched[k].append(v.assign(source_month=str(p)))
        # 당일 기체 일정: tail_*_hist 의 참조 데이터 (해당 월 BTS만 사용)
        df["sched_dep_utc"] = utc_departure(df, sm)
        t = pd.concat([df[["FlightDate"] + FE.TAIL_KEY], FE.tail_schedule_features(df)], axis=1)
        d = C.AGG / "tail_sameday" / f"year={p.year}" / f"month={p.month:02d}"
        d.mkdir(parents=True, exist_ok=True)
        t[TAIL_REF_COLS].assign(source_month=str(p)).to_parquet(d / "part.parquet", **C.PARQUET_KW)
        dist.append(df.groupby(["Origin", "Dest"])["Distance"].agg(lambda s: s.mode().iloc[0]).reset_index()
                    .assign(source_month=str(p)))
        print(f"  aggregates {p}: {len(df):,} rows", flush=True)

    targets = [f"{y}-{m:02d}" for y, m in C.dataset_months()]
    for name, parts in outcome.items():
        counts = pd.concat(parts, ignore_index=True)
        counts.to_parquet(C.AGG / f"monthly_outcomes_{name}.parquet", **C.PARQUET_KW)
        table = FE.rolling_hist_table(counts, name, targets, C.HIST_WINDOW_MONTHS, C.HIST_SMOOTH_M,
                                      lag=C.BTS_LAG_MONTHS)
        table["source_max_month"] = table["window_last_month"]
        table.to_parquet(C.AGG / f"{name}.parquet", **C.PARQUET_KW)
    for name, parts in sched.items():
        pd.concat(parts, ignore_index=True).to_parquet(C.AGG / f"sched_counts_{name}.parquet", **C.PARQUET_KW)
    # 노선 거리: 최신 월 기준 (서비스 입력에 거리가 없으므로 BTS에서 조회)
    pd.concat(dist, ignore_index=True).to_parquet(C.AGG / "route_distance.parquet", **C.PARQUET_KW)
    return n_rows


def read_sched(name, lo, hi):
    return pd.read_parquet(C.AGG / f"sched_counts_{name}.parquet",
                           filters=[("FlightDate", ">=", lo), ("FlightDate", "<=", hi)])


def read_tail_ref(periods):
    parts = []
    for p in periods:
        f = C.AGG / "tail_sameday" / f"year={p.year}" / f"month={p.month:02d}" / "part.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=TAIL_REF_COLS))
    t = pd.concat(parts, ignore_index=True)
    t["FlightDate"] = pd.to_datetime(t["FlightDate"])
    return t


class Stats:
    """샘플 전 BTS 전체 기준 검증 통계 (data_quality_report 용)."""

    def __init__(self):
        self.month_class = []          # split, ym, label, n_full, n_sample, weighted
        self.corr = {}                 # name -> [n, sx, sy, sxx, syy, sxy]
        self.timing_violations = Counter()
        self.id_mismatch = []          # 월별 v2 와 flight_id 집합 비교
        self.hist_missing = Counter()
        self.tail = []                 # split, label, n, n_found (샘플 기준)

    def add_corr(self, name, x, y):
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m].astype(float), y[m].astype(float)
        acc = self.corr.setdefault(name, [0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i, v in enumerate([len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]):
            acc[i] += float(v)

    def corr_values(self):
        out = {}
        for k, (n, sx, sy, sxx, syy, sxy) in self.corr.items():
            cov = sxy / n - sx / n * sy / n
            out[k] = round(cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2)), 4)
        return out


def process_month(year, month, split, stats):
    """BTS 쪽 피처만 다시 계산. 운항월 M 에 대해 M-BTS_LAG_MONTHS 월 말까지의 BTS만 사용."""
    mp = pd.Period(f"{year}-{month:02d}", freq="M")
    src = C.source_month(mp)
    src_end = src.end_time.normalize()
    df = read_month(year, month, ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin",
                                  "Dest", "CRSDepTime", "CRSArrTime", "Cancelled", "DepDelay"])
    df["flight_id"] = (year * 100 + month) * 10_000_000 + np.arange(len(df), dtype="int64")
    df = add_labels(df)

    # ---- 혼잡도 (BTS 전체 행): oracle(당일, 학습 제외) / 작년 같은 요일 / 공개된 최신 달 같은 요일
    m0, m1 = mp.start_time, mp.end_time.normalize()
    fmt = "%Y-%m-%d"
    oracle_counts = {k: read_sched(k, m0.strftime(fmt), m1.strftime(fmt)) for k in FE.CONGESTION_SPECS}
    ly_lo = (m0 - pd.Timedelta(days=max(C.PROXY_LY_LAGS_DAYS))).strftime(fmt)
    ly_hi = (m1 - pd.Timedelta(days=min(C.PROXY_LY_LAGS_DAYS))).strftime(fmt)
    ly_counts = {k: read_sched(k, ly_lo, ly_hi) for k in FE.CONGESTION_SPECS}
    src_counts = {k: read_sched(k, src.start_time.strftime(fmt), src_end.strftime(fmt)) for k in FE.CONGESTION_SPECS}
    oracle = FE.congestion_lag_features(df, oracle_counts, (0,), "oracle")
    ly = FE.congestion_lag_features(df, ly_counts, C.PROXY_LY_LAGS_DAYS, "proxy_ly")
    recent = FE.congestion_recent_features(df, src_counts, src)
    for name in FE.CONGESTION_SPECS:
        o = oracle[f"{name}_oracle"].to_numpy()
        stats.add_corr(f"{name}|proxy_ly", ly[f"{name}_proxy_ly"].to_numpy(), o)
        stats.add_corr(f"{name}|proxy_recent", recent[f"{name}_proxy_recent"].to_numpy(), o)
    df = pd.concat([df, ly, recent], axis=1)

    # ---- 샘플링 (v2와 같은 seed·규칙 → 같은 행)
    s = sample_month(df, split, year, month)
    v2_ids = pd.read_parquet(C.V2_MERGED / f"split={split}" / f"year={year}" / f"month={month:02d}" / "part.parquet",
                             columns=["flight_id"])["flight_id"]
    same = set(s["flight_id"]) == set(v2_ids)
    stats.id_mismatch.append(dict(ym=str(mp), split=split, v3=len(s), v2=len(v2_ids), identical=same))
    full_cnt = df["label"].value_counts()
    samp = s.groupby("label").agg(n=("label", "size"), w=("sample_weight", "sum"))
    for lab in sorted(full_cnt.index):
        stats.month_class.append(dict(split=split, ym=str(mp), label=int(lab), n_full=int(full_cnt[lab]),
                                      n_sample=int(samp["n"].get(lab, 0)), weighted=float(samp["w"].get(lab, 0))))
    del df

    # ---- 샘플된 행: 과거 실적(M-14~M-3), 작년 대비 추세, tail_*_hist
    s = s.reset_index(drop=True)
    s["dep_hour_local"] = FE.hhmm_to_minutes(s["CRSDepTime"]) % 1440 // 60
    s["hist_month"] = str(mp)
    src_dates = {"hist": pd.Series(pd.NaT, index=s.index)}
    for name, keys in FE.HIST_SPECS.items():
        h = pd.read_parquet(C.AGG / f"{name}.parquet", filters=[("hist_month", "==", str(mp))])
        h = h.drop(columns=["window_first_month", "window_last_month"]).rename(
            columns={"source_max_month": f"{name}_source_max_month"})
        s = s.merge(h, on=["hist_month"] + keys, how="left")
        stats.hist_missing[f"{split}|{name}"] += int(s[f"{name}_delay_rate"].isna().sum())
        last = pd.PeriodIndex(s[f"{name}_source_max_month"].fillna(str(src)), freq="M").end_time.normalize()
        stats.timing_violations[name] += int((last > src_end).sum())
        s = s.drop(columns=[f"{name}_source_max_month"])
    yoy = FE.origin_yoy_table(pd.concat([read_sched("origin_sched_deps_day", (src - 12).start_time.strftime(fmt),
                                                    (src - 12).end_time.strftime(fmt)),
                                         src_counts["origin_sched_deps_day"]]), src)
    s["origin_yoy_ratio"] = s["Origin"].map(yoy).astype(float)

    ref_periods = sorted({src} | {pd.Period(d, freq="M") for d in pd.date_range(
        m0 - pd.Timedelta(days=max(C.TAIL_LY_LAGS_DAYS)), m1 - pd.Timedelta(days=min(C.TAIL_LY_LAGS_DAYS)), freq="D")})
    th = FE.tail_hist_features(s, read_tail_ref(ref_periods), src, C.TAIL_LY_LAGS_DAYS)
    s = pd.concat([s, th], axis=1)
    stats.timing_violations["tail_hist"] += int((s["tail_ref_max_date"] > src_end).sum())
    ly_max = pd.to_datetime(s["FlightDate"]) - pd.Timedelta(days=min(C.PROXY_LY_LAGS_DAYS))
    stats.timing_violations["proxy_ly"] += int((ly_max > src_end).sum())
    stats.timing_violations["proxy_recent/origin_yoy"] += 0   # src 월만 읽음 (구조상 0)
    s["bts_source_max_date"] = pd.concat([ly_max, s["tail_ref_max_date"]], axis=1).max(axis=1).where(
        lambda x: x > src_end, src_end)
    s["bts_source_limit_date"] = src_end
    for lab, g in s.groupby("label"):
        stats.tail.append(dict(split=split, label=int(lab), n=len(g), n_found=int((g["tail_ref_days_found"] > 0).sum()),
                               weighted_n=float(g["sample_weight"].sum()),
                               weighted_found=float(g.loc[g["tail_ref_days_found"] > 0, "sample_weight"].sum())))

    keep = (["flight_id", "label", "hist_month", "origin_yoy_ratio", "bts_source_max_date", "bts_source_limit_date",
             "tail_ref_max_date"]
            + [c for c in s.columns if c.startswith("hist_") and c != "hist_month"]
            + [c for c in s.columns if c.endswith("_proxy_ly") or c.endswith("_proxy_recent")]
            + FE.TAIL_HIST_FEATURES)
    out = C.FLIGHT_INTERIM / f"split={split}" / f"year={year}" / f"month={month:02d}"
    out.mkdir(parents=True, exist_ok=True)
    s[keep].to_parquet(out / "part.parquet", **C.PARQUET_KW)
    return int(full_cnt.sum()), len(s), same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-aggregates", action="store_true", help="aggregates/ 가 이미 있으면 1단계 생략")
    ap.add_argument("--months", help="테스트용: 처리할 월 YYYY-MM 목록(쉼표)")
    a = ap.parse_args()
    timer = C.RunTimer("preprocess_flights")
    C.ensure_dirs()
    sm = load_station_map()
    n_agg = None
    if not a.skip_aggregates:
        n_agg = build_aggregates(sm)
    months = C.dataset_months()
    if a.months:
        months = [(int(x[:4]), int(x[5:7])) for x in a.months.split(",")]
    stats = Stats()
    tot_full = tot_s = 0
    for year, month in months:
        split = C.split_of_month(year, month)
        n_full, n_s, same = process_month(year, month, split, stats)
        tot_full += n_full
        tot_s += n_s
        print(f"  {year}-{month:02d} {split}: {n_full:,} -> {n_s:,} same_ids_as_v2={same} "
              f"(peak {C.peak_memory_mb()} MB)", flush=True)
    if a.months:
        return
    pd.DataFrame(stats.month_class).to_parquet(C.INTERIM / "sampling_stats.parquet", **C.PARQUET_KW)
    pd.DataFrame(stats.tail).to_parquet(C.INTERIM / "tail_hist_stats.parquet", **C.PARQUET_KW)
    pd.DataFrame(stats.id_mismatch).to_parquet(C.INTERIM / "v2_id_check.parquet", **C.PARQUET_KW)
    (C.INTERIM / "flight_stats.json").write_text(json.dumps(dict(
        congestion_corr=stats.corr_values(), timing_violations=dict(stats.timing_violations),
        hist_missing=dict(stats.hist_missing), bts_rows_aggregated=n_agg, flights_full=tot_full,
        flights_sampled=tot_s), indent=1, ensure_ascii=False), encoding="utf-8")
    timer.done(flights_full=tot_full, flights_sampled=tot_s)


if __name__ == "__main__":
    main()
