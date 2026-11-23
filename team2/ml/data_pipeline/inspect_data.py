"""BTS·IEM 원본 탐색 (읽기 전용). 결과: processed/inspection/*.csv, processed/data_inspection_report.md

전체 파일을 한 번씩 순회하며 정확한 통계(행 수, 결측, 최소·최대·평균, 범주 분포, 중복)를 집계하고,
분위수·IQR 이상값 비율은 파일마다 무작위 표본(BTS 2%, IEM 5%)을 모아 추정한다.

사용: python ml/data_pipeline/inspect_data.py [--bts-only | --weather-only]
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from feature_engineering import parse_precip  # noqa: E402

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------- 컬럼 설명 (공식 문서 기준)
# BTS: 원본 zip의 readme.html (TranStats Reporting Carrier On-Time Performance record layout)
BTS_DESC = {
    "Year": "연도", "Quarter": "분기(1-4)", "Month": "월", "DayofMonth": "일", "DayOfWeek": "요일(1=월 ... 7=일)",
    "FlightDate": "운항일(현지, yyyy-mm-dd)",
    "Reporting_Airline": "항공사 고유 코드(Unique Carrier Code)",
    "Flight_Number_Reporting_Airline": "편명 번호", "Tail_Number": "기체 등록번호",
    "OriginAirportID": "출발 공항 DOT ID", "Origin": "출발 공항 코드(IATA)", "OriginCityName": "출발 도시명",
    "OriginState": "출발 주 코드", "DestAirportID": "도착 공항 DOT ID", "Dest": "도착 공항 코드(IATA)",
    "DestCityName": "도착 도시명", "DestState": "도착 주 코드",
    "CRSDepTime": "예정 출발 시각(현지 hhmm)", "DepTimeBlk": "예정 출발 시간대 블록(1시간)",
    "CRSArrTime": "예정 도착 시각(현지 hhmm)", "ArrTimeBlk": "예정 도착 시간대 블록(1시간)",
    "CRSElapsedTime": "예정 비행 소요시간(분)", "Distance": "공항 간 거리(mile)",
    "DistanceGroup": "거리 구간(250mile 단위)",
    "DepTime": "실제 출발 시각(현지 hhmm)", "DepDelay": "출발 지연(분, 조기출발 음수)",
    "DepDelayMinutes": "출발 지연(분, 조기출발 0)", "DepDel15": "15분 이상 출발 지연 여부(1=예)",
    "DepartureDelayGroups": "출발 지연 구간(15분 단위)", "TaxiOut": "택시아웃 시간(분)",
    "WheelsOff": "이륙 시각(현지 hhmm)", "WheelsOn": "착륙 시각(현지 hhmm)", "TaxiIn": "택시인 시간(분)",
    "ArrTime": "실제 도착 시각(현지 hhmm)", "ArrDelay": "도착 지연(분, 조기도착 음수)",
    "ArrDelayMinutes": "도착 지연(분, 조기도착 0)", "ArrDel15": "15분 이상 도착 지연 여부(1=예)",
    "ArrivalDelayGroups": "도착 지연 구간(15분 단위)", "ActualElapsedTime": "실제 소요시간(분)",
    "AirTime": "비행 시간(분)", "Cancelled": "결항 여부(1=예)",
    "CancellationCode": "결항 사유(A=항공사, B=기상, C=NAS, D=보안)", "Diverted": "회항 여부(1=예)",
    "CarrierDelay": "항공사 원인 지연(분)", "WeatherDelay": "기상 원인 지연(분)", "NASDelay": "NAS 원인 지연(분)",
    "SecurityDelay": "보안 원인 지연(분)", "LateAircraftDelay": "선행 항공기 지연(분)",
}
# IEM: asos.py 도움말 + IEM ASOS 다운로드 페이지 단위 설명
IEM_DESC = {
    "station": "IEM 관측소 ID", "valid": "관측 시각(UTC)", "lon": "관측소 경도", "lat": "관측소 위도",
    "elevation": "관측소 고도(m)", "tmpf": "기온(°F)", "dwpf": "이슬점(°F)", "relh": "상대습도(%)",
    "feel": "체감온도(°F, 열지수/풍속냉각)", "drct": "풍향(도, 진북 기준)", "sknt": "풍속(knot)",
    "gust": "돌풍(knot)", "peak_wind_gust": "최대 순간풍(knot, 지난 1시간 RMK PK WND)",
    "vsby": "시정(statute mile)", "skyc1": "1층 운량(CLR/FEW/SCT/BKN/OVC/VV)", "skyc2": "2층 운량",
    "skyc3": "3층 운량", "skyc4": "4층 운량", "skyl1": "1층 운저 고도(ft AGL)", "skyl2": "2층 운저 고도(ft)",
    "skyl3": "3층 운저 고도(ft)", "skyl4": "4층 운저 고도(ft)",
    "p01i": "1시간 강수량(inch, 정시 관측 기준 누적; T=미량)", "wxcodes": "현재 일기 코드(METAR)",
    "ice_accretion_1hr": "1시간 착빙(inch)", "ice_accretion_3hr": "3시간 착빙(inch)",
    "ice_accretion_6hr": "6시간 착빙(inch)", "snowdepth": "적설 깊이(inch)", "alti": "고도계 설정치(inHg)",
    "mslp": "해면기압(hPa)", "metar": "원문 METAR",
}

BTS_STR = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Tail_Number", "Origin",
           "OriginCityName", "OriginState", "Dest", "DestCityName", "DestState", "CRSDepTime", "DepTimeBlk",
           "CRSArrTime", "ArrTimeBlk", "DepTime", "WheelsOff", "WheelsOn", "ArrTime", "CancellationCode"]
BTS_KEY = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime"]
BTS_CAT_FULL = ["Reporting_Airline", "Origin", "Dest", "OriginState", "DestState", "DepTimeBlk", "ArrTimeBlk",
                "CancellationCode", "DistanceGroup", "DepartureDelayGroups", "ArrivalDelayGroups", "Cancelled",
                "Diverted", "DepDel15", "ArrDel15", "Year", "Quarter", "Month", "DayOfWeek", "DayofMonth"]
BTS_UNIQUE_ONLY = ["Tail_Number", "Flight_Number_Reporting_Airline", "OriginCityName", "DestCityName",
                   "OriginAirportID", "DestAirportID", "CRSDepTime", "CRSArrTime"]
# 도메인 기준 이상값 (정확 집계)
BTS_RULES = {
    "CRSElapsedTime<=0": lambda d: d["CRSElapsedTime"] <= 0,
    "Distance<=0": lambda d: d["Distance"] <= 0,
    "DepDelay>=600": lambda d: d["DepDelay"] >= 600,
    "DepDelay<=-60": lambda d: d["DepDelay"] <= -60,
    "TaxiOut>=180": lambda d: d["TaxiOut"] >= 180,
    "AirTime>CRSElapsedTime*2": lambda d: d["AirTime"] > d["CRSElapsedTime"] * 2,
    "CRSDepTime not hhmm": lambda d: ~d["CRSDepTime"].fillna("").str.fullmatch(r"(?:[01]\d|2[0-4])[0-5]\d"),
    "Cancelled & DepTime present": lambda d: (d["Cancelled"] == 1) & d["DepTime"].notna(),
    "not Cancelled & DepDelay missing": lambda d: (d["Cancelled"] == 0) & d["DepDelay"].isna(),
}

IEM_NUM = ["lon", "lat", "elevation", "tmpf", "dwpf", "relh", "feel", "drct", "sknt", "gust", "peak_wind_gust",
           "vsby", "skyl1", "skyl2", "skyl3", "skyl4", "p01i", "ice_accretion_1hr", "ice_accretion_3hr",
           "ice_accretion_6hr", "snowdepth", "alti", "mslp"]
IEM_CAT = ["skyc1", "skyc2", "skyc3", "skyc4"]
IEM_RULES = {
    "tmpf<-60 or >130": lambda d: (d["tmpf"] < -60) | (d["tmpf"] > 130),
    "dwpf>tmpf+1": lambda d: d["dwpf"] > d["tmpf"] + 1,
    "relh>100": lambda d: d["relh"] > 100,
    "sknt>100": lambda d: d["sknt"] > 100,
    "gust<sknt": lambda d: d["gust"] < d["sknt"],
    "vsby>10": lambda d: d["vsby"] > 10,
    "p01i>4": lambda d: d["p01i"] > 4,
    "alti<27 or >32": lambda d: (d["alti"] < 27) | (d["alti"] > 32),
    "drct>360": lambda d: d["drct"] > 360,
}


class NumStats:
    def __init__(self):
        self.n = defaultdict(int)
        self.s = defaultdict(float)
        self.ss = defaultdict(float)
        self.mn = {}
        self.mx = {}

    def add(self, df, cols):
        for c in cols:
            v = df[c].dropna().to_numpy(dtype=float)
            if not len(v):
                continue
            self.n[c] += len(v)
            self.s[c] += v.sum()
            self.ss[c] += (v * v).sum()
            self.mn[c] = min(self.mn.get(c, np.inf), v.min())
            self.mx[c] = max(self.mx.get(c, -np.inf), v.max())

    def row(self, c, sample):
        n = self.n.get(c, 0)
        out = dict(count=n, min=self.mn.get(c), max=self.mx.get(c))
        if n:
            mean = self.s[c] / n
            out.update(mean=round(mean, 4), std=round(np.sqrt(max(self.ss[c] / n - mean ** 2, 0)), 4))
        v = sample[c].dropna().astype(float) if c in sample else pd.Series(dtype=float)
        if len(v):
            q = v.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            iqr = q[0.75] - q[0.25]
            lo, hi = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
            out.update(p01=q[0.01], p25=q[0.25], p50=q[0.5], p75=q[0.75], p99=q[0.99],
                       iqr_outlier_pct_sample=round(100 * ((v < lo) | (v > hi)).mean(), 3))
        return out


def write_csv(path, rows, cols=None):
    rows = list(rows)
    cols = cols or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def label_counts(df):
    canc = df["Cancelled"] == 1
    delayed = ~canc & (df["DepDelay"] >= C.DELAY_THRESHOLD_MIN)
    normal = ~canc & (df["DepDelay"] < C.DELAY_THRESHOLD_MIN)
    return dict(normal=int(normal.sum()), delayed=int(delayed.sum()), cancelled=int(canc.sum()),
                undefined=int((~canc & df["DepDelay"].isna()).sum()), diverted=int((df["Diverted"] == 1).sum()))


# ---------------------------------------------------------------- BTS
def inspect_bts():
    files = sorted(C.BTS_DIR.glob("bts_*.csv.gz"))
    stats, miss, cat, uniq = NumStats(), Counter(), defaultdict(Counter), defaultdict(set)
    rules, samples, file_rows = Counter(), [], []
    origin_pairs = defaultdict(set)
    by_year = defaultdict(Counter)
    dtypes, cols0, total = {}, None, 0
    for f in files:
        df = pd.read_csv(f, dtype={c: str for c in BTS_STR}, low_memory=False)
        cols0 = cols0 or list(df.columns)
        if list(df.columns) != cols0:
            print(f"!! column mismatch in {f.name}")
        for c in df.columns:
            dtypes.setdefault(c, set()).add(str(df[c].dtype))
        num_cols = [c for c in df.columns if c not in BTS_STR]
        total += len(df)
        miss.update(df.isna().sum().to_dict())
        stats.add(df, num_cols)
        for c in BTS_CAT_FULL:
            cat[c].update(df[c].fillna("<NA>").astype(str).tolist())
        for c in BTS_UNIQUE_ONLY:
            uniq[c].update(df[c].dropna().astype(str).unique().tolist())
        for name, fn in BTS_RULES.items():
            rules[name] += int(fn(df).sum())
        for a, b in df[["Origin", "OriginAirportID"]].drop_duplicates().itertuples(index=False):
            origin_pairs[a].add(b)
        for k, v in label_counts(df).items():
            by_year[f.name[4:8]][k] += v
        file_rows.append(dict(file=f.name, rows=len(df), cols=df.shape[1],
                              min_date=df["FlightDate"].min(), max_date=df["FlightDate"].max(),
                              dup_full_rows=int(df.duplicated().sum()),
                              dup_flight_key=int(df.duplicated(BTS_KEY).sum()),
                              size_mb=round(f.stat().st_size / 2 ** 20, 1)))
        samples.append(df.sample(frac=0.02, random_state=int(RNG.integers(1e9))))
        print(f"  {f.name}: {len(df):,} rows", flush=True)
    sample = pd.concat(samples, ignore_index=True)

    write_csv(C.INSPECTION / "bts_files.csv", file_rows)
    col_rows = []
    for c in cols0:
        r = dict(column=c, description=BTS_DESC.get(c, ""), dtype="/".join(sorted(dtypes[c])),
                 missing_n=int(miss[c]), missing_pct=round(100 * miss[c] / total, 3),
                 n_unique=len(cat[c]) if c in cat else (len(uniq[c]) if c in uniq else ""))
        if c not in BTS_STR:
            r.update(stats.row(c, sample))
        col_rows.append(r)
    write_csv(C.INSPECTION / "bts_columns.csv", col_rows,
              ["column", "description", "dtype", "missing_n", "missing_pct", "n_unique", "count", "min", "max",
               "mean", "std", "p01", "p25", "p50", "p75", "p99", "iqr_outlier_pct_sample"])
    cat_rows = [dict(column=c, value=v, count=n, pct=round(100 * n / total, 3))
                for c in BTS_CAT_FULL for v, n in cat[c].most_common()]
    write_csv(C.INSPECTION / "bts_categorical_counts.csv", cat_rows)
    write_csv(C.INSPECTION / "bts_rule_outliers.csv",
              [dict(rule=k, count=v, pct=round(100 * v / total, 4)) for k, v in rules.items()])
    ap = [dict(airport=a, origin_flights=cat["Origin"][a], dest_flights=cat["Dest"].get(a, 0))
          for a, _ in cat["Origin"].most_common()]
    write_csv(C.INSPECTION / "bts_airport_counts.csv", ap)
    yr = []
    for y, cnt in sorted(by_year.items()):
        n = sum(cnt[k] for k in ("normal", "delayed", "cancelled", "undefined"))
        yr.append(dict(year=y, flights=n, **cnt,
                       **{f"{k}_pct": round(100 * cnt[k] / n, 2) for k in ("normal", "delayed", "cancelled")}))
    write_csv(C.INSPECTION / "bts_class_ratio_by_year.csv", yr)

    # 중복/파생 관계 (표본 기준)
    s = sample
    rel = {
        "DepDelayMinutes == max(DepDelay,0)": (s["DepDelayMinutes"] == s["DepDelay"].clip(lower=0))[s["DepDelay"].notna()].mean(),
        "DepDel15 == (DepDelay>=15)": (s["DepDel15"] == (s["DepDelay"] >= 15).astype(float))[s["DepDelay"].notna()].mean(),
        "ArrDelayMinutes == max(ArrDelay,0)": (s["ArrDelayMinutes"] == s["ArrDelay"].clip(lower=0))[s["ArrDelay"].notna()].mean(),
        "DistanceGroup == min(Distance//250+1, 11)": (s["DistanceGroup"] == np.minimum(s["Distance"] // 250 + 1, 11)).mean(),
        "DepTimeBlk == hour(CRSDepTime) block": (s["DepTimeBlk"].str[:2] == s["CRSDepTime"].str[:2].replace("24", "00")).mean(),
        "Year/Month/DayofMonth == FlightDate": (pd.to_datetime(s["FlightDate"]) == pd.to_datetime(
            dict(year=s["Year"], month=s["Month"], day=s["DayofMonth"]))).mean(),
        "DayOfWeek == FlightDate weekday+1": (pd.to_datetime(s["FlightDate"]).dt.dayofweek + 1 == s["DayOfWeek"]).mean(),
        "Cancelled==1 -> DepDelay missing": s.loc[s["Cancelled"] == 1, "DepDelay"].isna().mean(),
        "CancellationCode present <-> Cancelled==1": (s["CancellationCode"].notna() == (s["Cancelled"] == 1)).mean(),
        "delay causes present <-> ArrDel15==1": (s["CarrierDelay"].notna() == (s["ArrDel15"] == 1)).mean(),
    }
    multi = {a: sorted(b) for a, b in origin_pairs.items() if len(b) > 1}
    write_csv(C.INSPECTION / "bts_column_relations.csv",
              [dict(relation=k, agreement_pct=round(100 * v, 3)) for k, v in rel.items()]
              + [dict(relation=f"Origin code with >1 OriginAirportID: {multi}", agreement_pct="")])
    print(f"BTS total rows {total:,}")
    return dict(total=total, files=len(files), cols=len(cols0))


# ---------------------------------------------------------------- IEM
def wx_tokens(s):
    toks = Counter()
    for v in s.dropna():
        for t in v.split():
            toks[re.sub(r"^[+-]|^VC", "", t)] += 1
    return toks


def inspect_weather():
    smap = pd.read_csv(C.STATION_MAP)
    files = sorted(C.ASOS_DIR.glob("*/*.csv.gz"))
    stats, miss, cat, rules, toks = NumStats(), Counter(), defaultdict(Counter), Counter(), Counter()
    samples, st_rows, total, cols0 = [], defaultdict(lambda: dict(rows=0, first="", last="")), 0, None
    dup_full = dup_key = speci_like = 0
    for i, f in enumerate(files):
        df = pd.read_csv(f, dtype=str)
        cols0 = cols0 or list(df.columns)
        total += len(df)
        miss.update(df.isna().sum().to_dict())
        for c in ("p01i", "ice_accretion_1hr", "ice_accretion_3hr", "ice_accretion_6hr", "snowdepth"):
            df[c] = parse_precip(df[c])
        for c in IEM_NUM:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        stats.add(df, IEM_NUM)
        for c in IEM_CAT:
            cat[c].update(df[c].fillna("<NA>").tolist())
        toks.update(wx_tokens(df["wxcodes"]))
        for name, fn in RULES_SAFE(IEM_RULES, df):
            rules[name] += fn
        dup_full += int(df.duplicated().sum())
        dup_key += int(df.duplicated(["station", "valid"]).sum())
        speci_like += int((df["valid"].str[14:16] != df["valid"].str[14:16].mode().iloc[0]).sum())
        iata = f.parent.name
        r = st_rows[iata]
        r["rows"] += len(df)
        r["first"] = min(filter(None, [r["first"], df["valid"].min()]))
        r["last"] = max(r["last"], df["valid"].max())
        samples.append(df.drop(columns=["metar"]).sample(frac=0.05, random_state=int(RNG.integers(1e9))))
        if i % 200 == 0:
            print(f"  weather {i}/{len(files)}", flush=True)
    sample = pd.concat(samples, ignore_index=True)

    col_rows = []
    for c in cols0:
        r = dict(column=c, description=IEM_DESC.get(c, ""), missing_n=int(miss[c]),
                 missing_pct=round(100 * miss[c] / total, 3),
                 n_unique=len(cat[c]) if c in cat else "")
        if c in IEM_NUM:
            r.update(stats.row(c, sample))
        col_rows.append(r)
    write_csv(C.INSPECTION / "iem_columns.csv", col_rows,
              ["column", "description", "missing_n", "missing_pct", "n_unique", "count", "min", "max", "mean",
               "std", "p01", "p25", "p50", "p75", "p99", "iqr_outlier_pct_sample"])
    write_csv(C.INSPECTION / "iem_categorical_counts.csv",
              [dict(column=c, value=v, count=n, pct=round(100 * n / total, 3))
               for c in IEM_CAT for v, n in cat[c].most_common()])
    write_csv(C.INSPECTION / "iem_wxcode_tokens.csv",
              [dict(token=t, count=n, pct_of_rows=round(100 * n / total, 4)) for t, n in toks.most_common(80)])
    write_csv(C.INSPECTION / "iem_rule_outliers.csv",
              [dict(rule=k, count=v, pct=round(100 * v / total, 4)) for k, v in rules.items()]
              + [dict(rule="duplicated full rows", count=dup_full, pct=round(100 * dup_full / total, 4)),
                 dict(rule="duplicated (station, valid)", count=dup_key, pct=round(100 * dup_key / total, 4)),
                 dict(rule="obs minute != station's routine minute (SPECI 등)", count=speci_like,
                      pct=round(100 * speci_like / total, 3))])
    sm = smap.set_index("bts_iata")
    write_csv(C.INSPECTION / "iem_station_counts.csv",
              [dict(bts_iata=k, iem_station=sm.loc[k, "iem_station"], rows=v["rows"], first_valid=v["first"],
                    last_valid=v["last"]) for k, v in sorted(st_rows.items(), key=lambda kv: -kv[1]["rows"])])
    print(f"IEM total rows {total:,}")
    return dict(total=total, files=len(files), cols=len(cols0))


def RULES_SAFE(rules, df):
    for name, fn in rules.items():
        yield name, int(fn(df).fillna(False).sum())


# ---------------------------------------------------------------- Markdown 보고서
def md_table(df, floatfmt="{:,.3f}"):
    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            if v.is_integer():
                return f"{int(v):,}" if abs(v) >= 10000 else str(int(v))
            return floatfmt.format(v).rstrip("0").rstrip(".")
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report():
    I = C.INSPECTION
    bf, bc = pd.read_csv(I / "bts_files.csv"), pd.read_csv(I / "bts_columns.csv")
    ic = pd.read_csv(I / "iem_columns.csv")
    bcat, icat = pd.read_csv(I / "bts_categorical_counts.csv"), pd.read_csv(I / "iem_categorical_counts.csv")
    ap, st = pd.read_csv(I / "bts_airport_counts.csv"), pd.read_csv(I / "iem_station_counts.csv")
    yr = pd.read_csv(I / "bts_class_ratio_by_year.csv")
    smap = pd.read_csv(C.STATION_MAP)
    iem_files = len(list(C.ASOS_DIR.glob("*/*.csv.gz")))
    iem_total = int(st["rows"].sum())
    bts_ap = set(ap["airport"]) | set(bcat.loc[bcat["column"] == "Dest", "value"])
    common = bts_ap & set(st["bts_iata"])
    bts_total = int(bf["rows"].sum())

    def topn(df, col, n=12):
        d = df[df["column"] == col].head(n)[["value", "count", "pct"]]
        return md_table(d)

    out = [
        "# 원본 데이터 탐색 보고서 (BTS On-Time Performance · IEM ASOS METAR)", "",
        "`ml/data_pipeline/inspect_data.py`가 원본 파일을 읽기만 해서 집계한 결과입니다. 원본은 변경하지 않았습니다.",
        "- 정확 집계: 행 수, 결측, 최소/최대/평균/표준편차, 범주 분포, 중복, 도메인 규칙 위반",
        "- 표본 추정: 분위수(p01~p99), IQR 이상값 비율 (파일마다 BTS 2%, IEM 5% 무작위 표본)",
        "- 세부 표: `processed/inspection/*.csv`", "",
        "## 1. 파일 형식·개수·크기", "",
        md_table(pd.DataFrame([
            dict(dataset="BTS", path="flight_expectation/flight_expectation/data/bts_selected/", format="CSV (gzip), 월별",
                 files=len(bf), rows=bts_total, columns=int(bf["cols"].iloc[0]),
                 period=f"{bf['min_date'].min()} ~ {bf['max_date'].max()} (현지 날짜)"),
            dict(dataset="IEM", path="weather/asos_raw/{IATA}/", format="CSV (gzip), 공항×연도",
                 files=iem_files, rows=iem_total, columns=len(ic),
                 period=f"{st['first_valid'].min()} ~ {st['last_valid'].max()} (UTC)")])),
        "",
        "- BTS 월별 파일 72개는 모두 같은 48개 컬럼입니다. 파일별 행 수는 `inspection/bts_files.csv`에 있습니다 "
        f"(최소 {bf['rows'].min():,}행 {bf.loc[bf['rows'].idxmin(), 'file']}, 최대 {bf['rows'].max():,}행 {bf.loc[bf['rows'].idxmax(), 'file']}).",
        "- 같은 폴더에 원본 zip(`data/raw_zip/`, 전체 110여 개 컬럼)과 2026-08 단월 파일(`bts_2026_08_selected.csv`, `raw_bts_2026_08/`)도 있습니다. "
        "2026-08은 기상 데이터(~2026-01-01) 범위 밖이라 이번 결합에 쓰지 않았습니다.",
        "- IEM 파일 2,314개 = 386개 공항 × 6년 − 2개 (PSE 2020·2021은 IEM에 데이터 없음).", "",
        "## 2. BTS 컬럼 구조·결측·분포", "",
        "설명은 BTS 원본 zip의 `readme.html`(TranStats record layout) 기준입니다.", "",
        md_table(bc[["column", "description", "dtype", "missing_pct", "n_unique", "min", "max", "mean", "p50", "p99",
                     "iqr_outlier_pct_sample"]]),
        "", "### 중복", "",
        f"- 완전 중복 행: {int(bf['dup_full_rows'].sum()):,}",
        f"- 항공편 키(FlightDate, 항공사, 편명, 출발, 도착, 예정출발) 중복: {int(bf['dup_flight_key'].sum()):,}",
        "- 정보가 중복되는 컬럼 (표본 기준 일치율):", "",
        md_table(pd.read_csv(I / "bts_column_relations.csv")),
        "", "### 도메인 규칙 이상값 (전체 정확 집계)", "", md_table(pd.read_csv(I / "bts_rule_outliers.csv")), "",
        "- `Cancelled & DepTime present`: 게이트를 떠난 뒤 결항된 편입니다. Label은 결항을 먼저 판정하므로 Class 2가 됩니다.",
        "- `DepDelay>=600`: 10시간 이상 지연이 0.13% 있습니다. 실제 장시간 지연으로 보고 Label 기준(≥15분)에서는 그대로 Class 1입니다.",
        "- 수치형 IQR 이상값 비율이 높은 지연 컬럼은 오른쪽 꼬리가 긴 분포라서 생깁니다. 오류로 보지 않습니다.", "",
        "### 범주형 분포 (상위)", "",
        "**Reporting_Airline**", "", topn(bcat, "Reporting_Airline", 18), "",
        "**CancellationCode** (A=항공사, B=기상, C=NAS, D=보안; 결항편에만 존재)", "", topn(bcat, "CancellationCode"), "",
        "**DepTimeBlk**", "", topn(bcat, "DepTimeBlk", 19), "",
        "## 3. 공항별 데이터 수", "",
        f"- BTS 출발 공항 {len(ap)}개, 도착 공항 {int((bcat['column'] == 'Dest').sum())}개, 합집합 {len(bts_ap)}개.",
        "- 상위 10개 / 하위 5개 출발 공항:", "",
        md_table(pd.concat([ap.head(10), ap.tail(5)])),
        "", "- IEM 관측 행 수 상위 5개 / 하위 5개 (AWOS는 20분 간격이라 행이 많음):", "",
        md_table(pd.concat([st.head(5), st.tail(5)])), "",
        "## 4. 정상·지연·결항 비율 (연도별, Label 기준과 동일: 결항 우선, 출발지연 ≥15분)", "",
        md_table(yr), "",
        "- 2020년은 코로나로 결항률 6.0%, 지연률 8.6%로 다른 해와 분포가 다릅니다.",
        "- 결항이 아닌데 DepDelay가 비어 Label을 정할 수 없는 편은 없었습니다(0건).", "",
        "## 5. IEM 컬럼 구조·결측·분포", "",
        "설명·단위는 IEM `asos.py` 도움말과 다운로드 페이지 기준입니다.", "",
        md_table(ic[["column", "description", "missing_pct", "n_unique", "min", "max", "mean", "p01", "p50", "p99",
                     "iqr_outlier_pct_sample"]]),
        "", "- 현상이 있을 때만 보고되는 컬럼(gust, wxcodes, 상층 운량, 착빙, 적설)은 빈 값이 '현상 없음'을 뜻하는 경우가 많습니다.",
        "- 물리적으로 불가능한 값이 소수 있습니다(예: 운저 525,000ft, 시정 34,006mi, 기온 145°F, 고도계 0·90inHg). 전처리에서 범위 QC로 NaN 처리합니다.", "",
        md_table(pd.read_csv(I / "iem_rule_outliers.csv")), "",
        "**운량(skyc1~2)**", "", md_table(icat[icat["column"].isin(["skyc1", "skyc2"])]), "",
        "**wxcodes 토큰 상위 25** (강도 +/-와 VC 제거)", "", md_table(pd.read_csv(I / "iem_wxcode_tokens.csv").head(25)), "",
        "## 6. 항공편·날씨의 공통 공항·공통 기간", "",
        f"- 공통 공항: BTS 공항 {len(bts_ap)}개 중 IEM 관측 파일이 있는 공항 {len(common)}개. 매핑은 `weather/station_map.csv` "
        f"({(smap['match_type'] == 'exact').sum()}개 exact, {(smap['match_type'] == 'faa_code').sum()}개 FAA/ICAO 코드 상이, 모두 공항 구내 관측소).",
        "- 공통 기간: BTS 2020-01-01 ~ 2025-12-31(현지) ⊂ IEM 2019-12-31 ~ 2026-01-01(UTC). 시차 여유 하루씩 포함.",
        "- 관측 공백이 큰 공항·연도(커버리지 80% 미만)는 `weather/coverage_report.csv`에 있습니다: PSE 2020~2022, BQN(야간 공백), "
        "WYS(5~11월만 운영), HDN 2025, GST 2024, GUF 2020, IFP 2025.",
    ]
    (C.INSPECTION.parent / "data_inspection_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote data_inspection_report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bts-only", action="store_true")
    ap.add_argument("--weather-only", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="집계는 건너뛰고 inspection/*.csv로 보고서만 생성")
    a = ap.parse_args()
    C.INSPECTION.mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        if not a.weather_only:
            inspect_bts()
        if not a.bts_only:
            inspect_weather()
    write_report()


if __name__ == "__main__":
    main()
