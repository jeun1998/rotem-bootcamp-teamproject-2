"""항공편·기상 파생 특징 정의 (순수 함수 모음).

원칙
- 원본 컬럼이 있고 물리적·운항적 의미가 분명한 것만 만든다.
- 관측이 매칭되지 않은 항공편의 기상 특징은 NaN으로 둔다 (정상 날씨로 채우지 않음).
- 매칭된 METAR에서 wxcodes가 비어 있으면 "보고된 일기 현상 없음"(0)으로 본다.
  이것은 결측이 아니라 METAR 규칙상 현상 그룹이 생략된 것이다. 관측 자체가 없으면 NaN이다.
"""
import re

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CEILING_LAYERS = {"BKN", "OVC", "VV"}
NO_CEILING_FT = 30000.0   # 운량은 관측됐지만 BKN/OVC/VV 층이 없을 때 (= 실링 없음) 쓰는 상한값


def parse_precip(s):
    """IEM 강수·착빙·적설 컬럼: 'T'(미량) -> 0.0001, 빈 값 -> NaN."""
    s = s.astype("string")
    return pd.to_numeric(s.replace("T", "0.0001"), errors="coerce")


# ---------------------------------------------------------------- 기상: 관측 1건 단위
_TOKEN_STRIP = re.compile(r"^(?:[+-]|VC)+")


def _has(wx, pattern):
    return wx.str.contains(pattern, regex=True, na=False)


def weather_obs_features(w):
    """관측 행 단위 특징. w: IEM 원본 컬럼(숫자 변환 후)을 가진 DataFrame."""
    out = pd.DataFrame(index=w.index)
    out["temp_dew_spread_f"] = w["tmpf"] - w["dwpf"]

    calm = w["sknt"] == 0
    rad = np.deg2rad(w["drct"])
    out["wind_dir_sin"] = np.where(calm, 0.0, np.sin(rad))
    out["wind_dir_cos"] = np.where(calm, 0.0, np.cos(rad))
    out.loc[w["sknt"].isna(), ["wind_dir_sin", "wind_dir_cos"]] = np.nan
    out["wind_max_kt"] = w[["sknt", "gust"]].max(axis=1, skipna=True)
    out.loc[w["sknt"].isna() & w["gust"].isna(), "wind_max_kt"] = np.nan
    out["gust_reported"] = w["gust"].notna().astype(float)
    out.loc[w["sknt"].isna(), "gust_reported"] = np.nan
    out["strong_wind"] = ((w["sknt"] >= 20) | (w["gust"] >= 25)).astype(float)
    out.loc[w["sknt"].isna(), "strong_wind"] = np.nan

    # 실링: BKN/OVC/VV 중 가장 낮은 층. 운량이 관측됐는데 해당 층이 없으면 NO_CEILING_FT
    ceil = pd.Series(np.nan, index=w.index)
    for i in range(1, 5):
        layer = w[f"skyc{i}"].isin(CEILING_LAYERS)
        ceil = np.fmin(ceil, w[f"skyl{i}"].where(layer))
    sky_obs = w["skyc1"].notna()
    out["ceiling_ft"] = np.where(ceil.notna(), ceil, np.where(sky_obs, NO_CEILING_FT, np.nan))
    out["low_ceiling"] = np.where(out["ceiling_ft"].isna(), np.nan, (out["ceiling_ft"] < 1000).astype(float))
    out["low_visibility"] = np.where(w["vsby"].isna(), np.nan, (w["vsby"] < 3).astype(float))

    # FAA 비행 기상 등급: 0=VFR, 1=MVFR, 2=IFR, 3=LIFR (실링·시정 중 나쁜 쪽)
    c, v = out["ceiling_ft"], w["vsby"]
    cat_c = np.select([c < 500, c < 1000, c <= 3000], [3, 2, 1], 0).astype(float)
    cat_v = np.select([v < 1, v < 3, v <= 5], [3, 2, 1], 0).astype(float)
    out["flight_category"] = np.where(c.isna() | v.isna(), np.nan, np.maximum(cat_c, cat_v))

    # 현재 일기 (wxcodes). 매칭된 관측에서 빈 값 = 보고된 현상 없음
    wx = w["wxcodes"].fillna("")
    wx_tokens = wx.str.split().apply(lambda ts: " ".join(_TOKEN_STRIP.sub("", t) for t in ts))
    out["wx_thunder"] = _has(wx_tokens, r"TS")
    out["wx_snow"] = _has(wx_tokens, r"SN|SG")
    out["wx_rain"] = _has(wx_tokens, r"RA|DZ")
    out["wx_freezing"] = _has(wx_tokens, r"FZ(?:RA|DZ)|PL|IC")
    out["wx_hail"] = _has(wx_tokens, r"GR|GS")
    out["wx_fog"] = _has(wx_tokens, r"FG")
    out["wx_mist_haze"] = _has(wx_tokens, r"\bBR\b|\bHZ\b|FU")
    out["wx_heavy"] = _has(wx, r"(?:^|\s)\+")
    out["precip_any"] = _has(wx_tokens, r"RA|DZ|SN|SG|PL|IC|GR|GS|UP")
    flag_cols = ["wx_thunder", "wx_snow", "wx_rain", "wx_freezing", "wx_hail", "wx_fog", "wx_mist_haze",
                 "wx_heavy", "precip_any"]
    out[flag_cols] = out[flag_cols].astype(float)

    # 결빙(제빙) 가능 조건: 어는 강수, 또는 0°C 이하에서 강수·안개
    out["icing_risk"] = (out["wx_freezing"].astype(bool)
                         | ((w["tmpf"] <= 32) & (out["precip_any"].astype(bool) | out["wx_fog"].astype(bool))))
    out["icing_risk"] = np.where(w["tmpf"].isna() & (out["wx_freezing"] == 0), np.nan,
                                 out["icing_risk"].astype(float))

    # 기상 악화 정도 (0~6): 뇌우, 어는 강수, 눈, 시정<1mi, 실링<500ft, 강풍
    sev = pd.concat([out["wx_thunder"], out["wx_freezing"], out["wx_snow"],
                     (w["vsby"] < 1).astype(float).where(w["vsby"].notna()),
                     (out["ceiling_ft"] < 500).astype(float).where(out["ceiling_ft"].notna()),
                     out["strong_wind"]], axis=1)
    out["weather_severity"] = sev.sum(axis=1, min_count=1)
    return out


def weather_window_features(w, hours=3):
    """관측소별 직전 N시간(관측 시각 포함, 과거 방향만) 창 특징. w는 station, valid, is_routine 등을 가짐."""
    res = []
    for sid, g in w.groupby("station", sort=False):
        g = g.sort_values("valid").set_index("valid")
        r = g.rolling(f"{hours}h", closed="both")
        f = pd.DataFrame(index=g.index)
        f["station"] = sid
        f[f"thunder_{hours}h"] = r["wx_thunder"].max()
        f[f"snow_{hours}h"] = r["wx_snow"].max()
        f[f"freezing_{hours}h"] = r["wx_freezing"].max()
        f[f"max_wind_{hours}h_kt"] = r["wind_max_kt"].max()
        f[f"min_vsby_{hours}h"] = r["vsby"].min()
        f[f"min_ceiling_{hours}h_ft"] = r["ceiling_ft"].min()
        # 정시 관측의 1시간 강수만 합산 (SPECI의 p01i는 정시 이후 누적이라 중복 합산 방지)
        f[f"precip_{hours}h_in"] = g["p01i"].where(g["is_routine"]).rolling(f"{hours}h", closed="both").sum()
        f[f"n_special_obs_{hours}h"] = (~g["is_routine"]).astype(float).rolling(f"{hours}h", closed="both").sum()
        # 기압 경향: 현재 alti - N시간 전(±30분 이내 가장 가까운 과거 관측) alti
        past = pd.merge_asof(pd.DataFrame({"t": g.index - pd.Timedelta(hours=hours)}),
                             g[["alti"]].reset_index().rename(columns={"valid": "t", "alti": "alti_past"}),
                             on="t", direction="backward", tolerance=pd.Timedelta(minutes=30))
        f[f"alti_change_{hours}h"] = g["alti"].to_numpy() - past["alti_past"].to_numpy()
        res.append(f.reset_index())
    return pd.concat(res, ignore_index=True)


# ---------------------------------------------------------------- 항공편
def hhmm_to_minutes(s):
    v = pd.to_numeric(s, errors="coerce")
    return (v // 100) * 60 + v % 100


def season_of(month):
    return month.map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})


def flight_calendar_features(f):
    out = pd.DataFrame(index=f.index)
    dep_min = hhmm_to_minutes(f["CRSDepTime"]) % 1440
    arr_min = hhmm_to_minutes(f["CRSArrTime"]) % 1440
    out["dep_hour_local"] = dep_min // 60
    out["dep_minute_of_day"] = dep_min
    out["arr_hour_local"] = arr_min // 60
    out["overnight_arrival"] = (hhmm_to_minutes(f["CRSDepTime"]) + f["CRSElapsedTime"] >= 1440).astype(float)
    date = pd.to_datetime(f["FlightDate"])
    out["month"] = date.dt.month
    out["season"] = season_of(date.dt.month)
    out["day_of_week"] = f["DayOfWeek"]
    out["is_weekend"] = f["DayOfWeek"].isin([6, 7]).astype(int)
    out["day_of_month"] = date.dt.day
    # 30일 넘게 떨어진 공휴일은 모두 30으로 자르므로, ±60일 안의 공휴일만 보면 충분 (한 편만 넣어도 같은 값)
    hol = USFederalHolidayCalendar().holidays(date.min() - pd.Timedelta(days=60),
                                              date.max() + pd.Timedelta(days=60))
    hol = np.array(hol.values, dtype="datetime64[D]")
    d = date.values.astype("datetime64[D]")
    if len(hol) == 0:
        dist = np.full(len(d), 30)
    else:
        idx = np.searchsorted(hol, d)
        prev = hol[np.clip(idx - 1, 0, len(hol) - 1)]
        nxt = hol[np.clip(idx, 0, len(hol) - 1)]
        dist = np.minimum(np.abs((d - prev).astype(int)), np.abs((nxt - d).astype(int)))
    out["days_to_federal_holiday"] = np.minimum(dist, 30)
    out["route"] = f["Origin"] + "_" + f["Dest"]
    return out


# ---------------------------------------------------------------- 혼잡도 (예정 편수)
# 이름: (그룹 키, 시각 컬럼). 시각 컬럼이 있으면 그 현지 '시' 단위로 센다
CONGESTION_SPECS = {
    "origin_sched_deps_hour": (["Origin"], "dep_hour_local"),
    "origin_sched_deps_day": (["Origin"], None),
    "dest_sched_arrs_hour": (["Dest"], "arr_hour_local"),
    "carrier_sched_deps_origin_day": (["Origin", "Reporting_Airline"], None),
}


def daily_schedule_counts(f):
    """BTS 전체 행(결항 포함 = 계획 기준)으로 날짜별 예정 편수 표. 반환: {name: DataFrame(date, keys, hour?, n)}."""
    out = {}
    for name, (keys, hour_col) in CONGESTION_SPECS.items():
        cols = ["FlightDate"] + keys + ([hour_col] if hour_col else [])
        out[name] = f.groupby(cols, observed=True).size().rename("n").reset_index()
    return out


def congestion_features(f, counts):
    """당일 값(*_oracle)과 직전 4주 같은 요일 평균(*_proxy).

    counts: daily_schedule_counts 결과(여러 달을 이어 붙인 것). 해당 날짜에 편이 없으면 0편으로 본다.
    """
    import config as C
    out = pd.DataFrame(index=f.index)
    date = pd.to_datetime(f["FlightDate"])
    for name, (keys, hour_col) in CONGESTION_SPECS.items():
        cols = keys + ([hour_col] if hour_col else [])
        tbl = counts[name].copy()
        tbl["FlightDate"] = pd.to_datetime(tbl["FlightDate"])
        tbl = tbl.set_index(["FlightDate"] + cols)["n"]
        left = pd.concat([date.rename("FlightDate"), f[cols]], axis=1)
        out[f"{name}_oracle"] = tbl.reindex(pd.MultiIndex.from_frame(left)).to_numpy()
        lagged = []
        for k in C.CONGESTION_LAGS_DAYS:
            lk = left.assign(FlightDate=left["FlightDate"] - pd.Timedelta(days=k))
            lagged.append(np.nan_to_num(tbl.reindex(pd.MultiIndex.from_frame(lk)).to_numpy(dtype=float), nan=0.0))
        out[f"{name}_proxy"] = np.mean(lagged, axis=0)
    return out


def _count_lookup(tbl, keys_frame):
    """tbl: (FlightDate(datetime), keys..., n) 를 MultiIndex로, keys_frame 행마다 편수(없으면 0)."""
    v = tbl.reindex(pd.MultiIndex.from_frame(keys_frame)).to_numpy(dtype=float)
    return np.nan_to_num(v, nan=0.0)


def congestion_lag_features(f, counts, lags, suffix):
    """D-lag 날짜들의 편수 평균 (편이 없던 날은 0). counts: {name: daily counts(FlightDate str)}."""
    out = pd.DataFrame(index=f.index)
    date = pd.to_datetime(f["FlightDate"])
    for name, (keys, hour_col) in CONGESTION_SPECS.items():
        cols = keys + ([hour_col] if hour_col else [])
        tbl = counts[name].copy()
        tbl["FlightDate"] = pd.to_datetime(tbl["FlightDate"])
        tbl = tbl.set_index(["FlightDate"] + cols)["n"]
        left = pd.concat([date.rename("FlightDate"), f[cols]], axis=1)
        vals = [_count_lookup(tbl, left.assign(FlightDate=left["FlightDate"] - pd.Timedelta(days=k))) for k in lags]
        out[f"{name}_{suffix}"] = np.mean(vals, axis=0)
    return out


def congestion_recent_features(f, counts_src, src_period):
    """공개된 최신 달(src_period = M-3) 중 운항일과 같은 요일인 날들의 평균 편수 (편이 없던 날은 0)."""
    out = pd.DataFrame(index=f.index)
    days = pd.date_range(src_period.start_time, src_period.end_time.normalize(), freq="D")
    n_wd = pd.Series(days.dayofweek).value_counts()          # 요일별 날짜 수
    wd = pd.to_datetime(f["FlightDate"]).dt.dayofweek
    for name, (keys, hour_col) in CONGESTION_SPECS.items():
        cols = keys + ([hour_col] if hour_col else [])
        t = counts_src[name].copy()
        t["wd"] = pd.to_datetime(t["FlightDate"]).dt.dayofweek
        s = t.groupby(cols + ["wd"], observed=True)["n"].sum()
        s = s / s.index.get_level_values("wd").map(n_wd).to_numpy()
        left = pd.concat([f[cols], wd.rename("wd")], axis=1)
        out[f"{name}_proxy_recent"] = np.nan_to_num(
            s.reindex(pd.MultiIndex.from_frame(left)).to_numpy(dtype=float), nan=0.0)
    return out


def origin_yoy_table(day_counts, src_period):
    """출발 공항의 (src 월 편수) / (1년 전 같은 월 편수). day_counts: origin_sched_deps_day 일별 편수."""
    ym = day_counts["FlightDate"].str[:7]
    cur = day_counts[ym == str(src_period)].groupby("Origin")["n"].sum()
    prev = day_counts[ym == str(src_period - 12)].groupby("Origin")["n"].sum()
    r = (cur / prev.reindex(cur.index)).replace([np.inf, -np.inf], np.nan)
    return r.rename("origin_yoy_ratio")


TAIL_HIST_FEATURES = ["tail_leg_index_hist", "tail_sched_turn_min_hist", "tail_prev_origin_hist",
                      "tail_prev_sched_elapsed_hist", "tail_ref_days_found"]
TAIL_KEY = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin"]


def tail_reference_dates(dates, src_period, ly_lags):
    """운항일별 참조일: 작년 같은 요일(D-364/371/357) + src 월 중 같은 요일인 날들. 반환: (FlightDate, ref_date)."""
    d = pd.Series(pd.to_datetime(pd.Series(dates).unique()), name="FlightDate").astype("datetime64[ns]")
    ly = pd.concat([pd.DataFrame({"FlightDate": d, "ref_date": d - pd.Timedelta(days=k)}) for k in ly_lags])
    src_days = pd.Series(pd.date_range(src_period.start_time, src_period.end_time.normalize(), freq="D"))
    rec = pd.DataFrame({"FlightDate": d, "wd": d.dt.dayofweek}).merge(
        pd.DataFrame({"ref_date": src_days, "wd": src_days.dt.dayofweek}), on="wd").drop(columns="wd")
    out = pd.concat([ly, rec], ignore_index=True)
    return out.astype({"FlightDate": "datetime64[ns]", "ref_date": "datetime64[ns]"})


def tail_hist_features(f, ref_table, src_period, ly_lags):
    """같은 항공사+편명+출발 공항이 참조일에 배정된 기체의 하루 일정으로 계산한 tail 특징(중앙값, 출발지는 최빈값).

    ref_table: 참조일들의 당일 기체 일정 (FlightDate, TAIL_KEY, tail_leg_index_of_day, tail_sched_turn_min,
               tail_prev_origin, tail_prev_sched_elapsed). 당일(운항일) Tail_Number는 쓰지 않는다.
    반환: f.index 정렬, TAIL_HIST_FEATURES + tail_ref_max_date
    """
    refs = tail_reference_dates(f["FlightDate"], src_period, ly_lags)
    left = pd.DataFrame({"row": np.arange(len(f)),
                         "FlightDate": pd.to_datetime(f["FlightDate"]).astype("datetime64[ns]").to_numpy()})
    for k in TAIL_KEY:
        left[k] = f[k].to_numpy()
    long = left.merge(refs, on="FlightDate")
    rt = ref_table.rename(columns={"FlightDate": "ref_date"})
    rt["ref_date"] = pd.to_datetime(rt["ref_date"]).astype("datetime64[ns]")
    rt = rt[rt["tail_leg_index_of_day"].notna()].drop_duplicates(["ref_date"] + TAIL_KEY)
    long = long.merge(rt, on=["ref_date"] + TAIL_KEY, how="inner")
    g = long.groupby("row")
    out = pd.DataFrame(index=np.arange(len(f)))
    out["tail_leg_index_hist"] = g["tail_leg_index_of_day"].median()
    out["tail_sched_turn_min_hist"] = g["tail_sched_turn_min"].median()
    out["tail_prev_sched_elapsed_hist"] = g["tail_prev_sched_elapsed"].median()
    po = long.dropna(subset=["tail_prev_origin"])
    if len(po):
        mode = (po.groupby(["row", "tail_prev_origin"]).size().reset_index(name="c")
                .sort_values(["row", "c", "tail_prev_origin"], ascending=[True, False, True])
                .drop_duplicates("row").set_index("row")["tail_prev_origin"])
        out["tail_prev_origin_hist"] = mode
    else:
        out["tail_prev_origin_hist"] = pd.NA
    out["tail_prev_origin_hist"] = out["tail_prev_origin_hist"].astype("string")
    out["tail_ref_days_found"] = g.size().reindex(out.index).fillna(0).astype(float)
    out["tail_ref_max_date"] = g["ref_date"].max()
    out.index = f.index
    return out


# ---------------------------------------------------------------- 과거 실적 (월 단위 창)
HIST_SPECS = {"hist_carrier": ["Reporting_Airline"], "hist_origin": ["Origin"], "hist_dest": ["Dest"],
              "hist_route": ["Origin", "Dest"], "hist_origin_hour": ["Origin", "dep_hour_local"],
              "hist_carrier_origin": ["Reporting_Airline", "Origin"]}


def monthly_outcome_counts(f):
    """한 달치 BTS 전체 행에서 그룹×월별 편수·결항·운항·지연 수. 반환: {name: DataFrame}."""
    d = pd.DataFrame({
        "ym": f["FlightDate"].str[:7],
        "is_cancel": (f["Cancelled"] == 1).astype("int32"),
        "is_flown": ((f["Cancelled"] == 0) & f["DepDelay"].notna()).astype("int32"),
    })
    d["is_delay"] = ((d["is_flown"] == 1) & (f["DepDelay"] >= 15)).astype("int32")
    d = pd.concat([d, f[["Reporting_Airline", "Origin", "Dest", "dep_hour_local"]]], axis=1)
    return {name: d.groupby(["ym"] + keys, observed=True)
                   .agg(n=("is_cancel", "size"), canc=("is_cancel", "sum"), flown=("is_flown", "sum"),
                        dly=("is_delay", "sum")).reset_index()
            for name, keys in HIST_SPECS.items()}


def rolling_hist_table(counts, name, target_months, window=12, m=50, lag=1):
    """운항월 T 의 피처 = (T-lag-window+1) ~ (T-lag) 달력월의 실적, 베이지안 평활(prior = 같은 창의 전체 비율).

    v2: lag=1 (직전 12개월). v3: lag=BTS_LAG_MONTHS (공개된 BTS만).
    반환 컬럼: hist_month(T, 'YYYY-MM'), keys, {name}_delay_rate, _cancel_rate, _n_flights,
              window_first_month, window_last_month(= source_max_month)
    """
    keys = HIST_SPECS[name]
    counts = counts.copy()
    counts["p"] = pd.PeriodIndex(counts["ym"], freq="M")
    rows = []
    for t in target_months:
        tp = pd.Period(t, freq="M")
        last, first = tp - lag, tp - lag - window + 1
        w = counts[(counts["p"] >= first) & (counts["p"] <= last)]
        if w.empty:
            continue
        p_c = w["canc"].sum() / w["n"].sum()
        p_d = w["dly"].sum() / w["flown"].sum()
        g = w.groupby(keys, observed=True)[["n", "canc", "flown", "dly"]].sum().reset_index()
        g[f"{name}_cancel_rate"] = (g["canc"] + m * p_c) / (g["n"] + m)
        g[f"{name}_delay_rate"] = (g["dly"] + m * p_d) / (g["flown"] + m)
        g[f"{name}_n_flights"] = g["n"]
        g["hist_month"] = str(tp)
        g["window_first_month"] = str(first)
        g["window_last_month"] = str(last)
        rows.append(g.drop(columns=["n", "canc", "flown", "dly"]))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 기체 일정 (예정 시각만 사용)
TAIL_FEATURES = ["tail_leg_index_of_day", "tail_sched_turn_min", "tail_prev_origin", "tail_prev_sched_elapsed"]


def tail_schedule_features(f):
    """같은 날(FlightDate) 같은 기체의 예정 운항을 예정 출발 UTC 순으로 정렬해 만든 특징.

    앞 편의 '예정' 도착 = 앞 편 예정 출발 UTC + 예정 소요시간. 실제 시각·지연은 쓰지 않는다.
    Tail_Number가 없는 편은 모든 값이 NaN (결측 플래그는 만들지 않음).
    """
    out = pd.DataFrame(index=f.index, columns=TAIL_FEATURES, dtype=float)
    out["tail_prev_origin"] = pd.Series(pd.NA, index=f.index, dtype="string")
    tail = f["Tail_Number"].astype("string").str.strip()
    ok = tail.notna() & (tail != "") & f["sched_dep_utc"].notna()
    g = pd.DataFrame({"tail": tail[ok], "date": f.loc[ok, "FlightDate"], "dep": f.loc[ok, "sched_dep_utc"],
                      "elapsed": f.loc[ok, "CRSElapsedTime"], "origin": f.loc[ok, "Origin"]})
    g = g.sort_values(["tail", "date", "dep"])
    grp = g.groupby(["tail", "date"], sort=False)
    prev_dep, prev_el = grp["dep"].shift(), grp["elapsed"].shift()
    prev_arr = prev_dep + pd.to_timedelta(prev_el, unit="m")
    out.loc[g.index, "tail_leg_index_of_day"] = (grp.cumcount() + 1).astype(float)
    out.loc[g.index, "tail_sched_turn_min"] = (g["dep"] - prev_arr).dt.total_seconds() / 60
    out.loc[g.index, "tail_prev_sched_elapsed"] = prev_el
    out.loc[g.index, "tail_prev_origin"] = grp["origin"].shift().astype("string")
    return out
