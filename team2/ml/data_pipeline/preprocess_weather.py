"""IEM ASOS 관측 전처리: 숫자 변환 + QC + 관측 단위·직전 3시간 창 특징.

v2: 관측소 WX_STATION_BATCH개씩 전체 기간을 한 번에 처리(연말·연초에도 3시간 창이 끊기지 않음)하고
    processed_v2/interim/weather/year=YYYY/month=MM/batch-XX.parquet 로 바로 써서 메모리를 비운다.
사용: python ml/data_pipeline/preprocess_weather.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import feature_engineering as FE  # noqa: E402

NUM = ["lon", "lat", "elevation", "tmpf", "dwpf", "relh", "feel", "drct", "sknt", "gust", "peak_wind_gust",
       "vsby", "skyl1", "skyl2", "skyl3", "skyl4", "alti", "mslp"]
PRECIP = ["p01i", "ice_accretion_1hr", "ice_accretion_3hr", "ice_accretion_6hr", "snowdepth"]
# 물리적으로 불가능한 값 -> NaN (미국 관측 극값 기준의 넉넉한 범위)
QC_RANGES = {"tmpf": (-80, 135), "dwpf": (-100, 95), "relh": (0, 100.5), "drct": (0, 360), "sknt": (0, 120),
             "gust": (0, 180), "peak_wind_gust": (0, 200), "vsby": (0, 100), "alti": (27.0, 32.0),
             "mslp": (900, 1085), "p01i": (0, 10)}
# 현지 날짜 -> UTC 차이(최대 ±14h) + 예측 리드 3h + 허용 2h + 창 3h 를 덮는 여유
MARGIN = pd.Timedelta(days=C.WX_MONTH_MARGIN_DAYS + 3)
RAW_COLS = ["station", "valid", "tmpf", "dwpf", "relh", "feel", "drct", "sknt", "gust", "peak_wind_gust", "vsby",
            "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "p01i", "wxcodes",
            "ice_accretion_1hr", "ice_accretion_3hr", "ice_accretion_6hr", "snowdepth", "alti", "mslp", "metar"]
NUM = [c for c in NUM if c not in ("lon", "lat", "elevation")]


def load_batch(pairs, t0, t1):
    parts = []
    for iata, _ in pairs:
        for y in range(t0.year, t1.year + 1):
            f = C.ASOS_DIR / iata / f"{iata}_{y}.csv.gz"
            if f.exists():
                d = pd.read_csv(f, dtype=str, usecols=RAW_COLS)
                d["bts_iata"] = iata
                parts.append(d)
    w = pd.concat(parts, ignore_index=True)
    w["valid"] = pd.to_datetime(w["valid"], format="%Y-%m-%d %H:%M")
    return w[(w["valid"] >= t0) & (w["valid"] <= t1)]


def process(w, qc):
    """QC + 특징. w: 한 묶음의 관측 원본(str). qc: 관측값 무효 처리 수를 누적할 dict."""
    n_raw = len(w)
    # 같은 관측소·시각 중복은 마지막 행(정정 보고 COR 가능성)을 남김
    w = w.drop_duplicates(["station", "valid"], keep="last").sort_values(["station", "valid"]).reset_index(drop=True)
    qc["duplicates_removed"] = qc.get("duplicates_removed", 0) + n_raw - len(w)

    for c in NUM:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    for c in PRECIP:
        w[c] = FE.parse_precip(w[c])

    def add(k, v):
        qc[k] = qc.get(k, 0) + int(v)

    for c, (lo, hi) in QC_RANGES.items():
        bad = w[c].notna() & ~w[c].between(lo, hi)
        add(c, bad.sum())
        w.loc[bad, c] = float("nan")
    for i in range(1, 5):   # 운저 상한을 넘는 층은 고도만 무효 처리
        bad = w[f"skyl{i}"] > FE.NO_CEILING_FT
        add(f"skyl{i}", bad.sum())
        w.loc[bad, f"skyl{i}"] = float("nan")
    # 순간 튐(spike): 직전 6시간(현재 관측 제외, 과거만) 중앙값과 30°F 넘게 차이 나면 무효
    for c in ("tmpf", "dwpf"):
        med = (w.set_index("valid").groupby("station")[c]
               .rolling("6h", closed="left", min_periods=3).median().to_numpy())
        bad = (w[c] - med).abs() > 30
        add(f"{c}_spike", bad.sum())
        w.loc[bad, c] = float("nan")
    # 정시 관측 = 관측소별 최빈 관측 분(minute). 그 외(SPECI 등)는 특별 관측으로 간주
    minute = w["valid"].dt.minute
    routine_min = minute.groupby(w["station"]).transform(lambda m: m.mode().iloc[0])
    w["is_routine"] = minute == routine_min

    w = pd.concat([w, FE.weather_obs_features(w)], axis=1)
    win = FE.weather_window_features(w[["station", "valid", "is_routine", "wx_thunder", "wx_snow", "wx_freezing",
                                        "wind_max_kt", "vsby", "ceiling_ft", "p01i", "alti"]])
    w = w.merge(win, on=["station", "valid"], how="left")
    f64 = w.select_dtypes("float64").columns
    w[f64] = w[f64].astype("float32")
    return w


def main():
    timer = C.RunTimer("preprocess_weather")
    C.ensure_dirs()
    starts = [pd.Timestamp(a) for a, _ in C.SPLITS.values()]
    ends = [pd.Timestamp(b) for _, b in C.SPLITS.values()]
    t0, t1 = min(starts) - MARGIN, max(ends) + pd.Timedelta(days=1) + MARGIN
    keep0, keep1 = min(starts) - pd.Timedelta(days=C.WX_MONTH_MARGIN_DAYS + 1), t1
    sm = pd.read_csv(C.STATION_MAP, dtype=str)
    pairs = list(zip(sm["bts_iata"], sm["iem_station"]))
    qc, n_rows = {}, 0
    for b in range(0, len(pairs), C.WX_STATION_BATCH):
        w = process(load_batch(pairs[b:b + C.WX_STATION_BATCH], t0, t1), qc)
        w = w[(w["valid"] >= keep0) & (w["valid"] <= keep1)]
        n_rows += len(w)
        for (y, m), part in w.groupby([w["valid"].dt.year, w["valid"].dt.month]):
            d = C.WX_INTERIM / f"year={y}" / f"month={m:02d}"
            d.mkdir(parents=True, exist_ok=True)
            part.to_parquet(d / f"batch-{b // C.WX_STATION_BATCH:02d}.parquet", **C.PARQUET_KW)
        print(f"  batch {b // C.WX_STATION_BATCH}: {len(w):,} obs  (peak {C.peak_memory_mb()} MB)", flush=True)
        del w
    pd.DataFrame([qc]).to_parquet(C.INTERIM / "weather_qc_counts.parquet", **C.PARQUET_KW)
    print("QC set to NaN:", {k: v for k, v in qc.items() if v})
    timer.done(rows=n_rows, period=f"{keep0} ~ {keep1}")


def load_weather(start, end, columns=None):
    """[start, end] (UTC) 구간의 기상 특징을 Parquet에서 읽는다. 월 경계를 넘으면 인접 월 파티션도 읽는다."""
    parts = []
    for p in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        d = C.WX_INTERIM / f"year={p.year}" / f"month={p.month:02d}"
        for f in sorted(d.glob("*.parquet")):
            parts.append(pd.read_parquet(f, columns=columns))
    w = pd.concat(parts, ignore_index=True)
    w = w[(w["valid"] >= pd.Timestamp(start)) & (w["valid"] <= pd.Timestamp(end))]
    return w.sort_values(["station", "valid"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
