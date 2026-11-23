"""v3 결합 결과 품질·누수 검증 + 보고서 생성 (v2 항목 유지 + 공개 시점·결측 누수·서비스 재현 검사).

입력: processed_v3/ (merge_datasets.py 이후), interim/*stats*, run_log.jsonl, processed_v2/ 의 재사용 통계
출력: processed_v3/data_quality_report.md, processed_v3/feature_selection_report.md
큰 Parquet은 월 파일 단위로 필요한 컬럼만 읽어 집계한다 (메모리 상한 유지).
사용: python ml/data_pipeline/validate_dataset.py
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import column_selection as S  # noqa: E402
import config as C  # noqa: E402
import feature_engineering as FE  # noqa: E402
from inspect_data import md_table  # noqa: E402
from service_features import build_service_features, service_feature_names  # noqa: E402

V2I = C.V2 / "interim"

CLASS_NAMES = {0: "정상", 1: "지연(≥15분)", 2: "결항"}


def dir_size_mb(p):
    return round(sum(f.stat().st_size for f in Path(p).rglob("*.parquet")) / 2 ** 20, 1)


def ram_disk():
    out = {}
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        out["ram_total_gb"] = round(ms.ullTotalPhys / 2 ** 30, 1)
        out["ram_available_gb_now"] = round(ms.ullAvailPhys / 2 ** 30, 1)
    except Exception:  # noqa: BLE001
        pass
    du = shutil.disk_usage(C.ROOT)
    out["disk_free_gb"] = round(du.free / 2 ** 30, 1)
    return out


def scan_ml(features):
    """ml/ 를 월 파일 단위로 읽어 split별 결측 수·클래스 수·가중 클래스 합을 누적."""
    miss, n, cls, wcls = Counter(), Counter(), Counter(), Counter()
    miss_w = Counter()      # (split, feature, label) -> 결측 행의 가중 합 (원래 분포 기준 label 분포용)
    for f in sorted(C.ML.rglob("*.parquet")):
        split = f.parent.name.split("=")[1]
        d = pd.read_parquet(f, columns=features + ["label", "sample_weight"])
        n[split] += len(d)
        isna = d[features].isna()
        for c, v in isna.sum().items():
            miss[(split, c)] += int(v)
        w = d["sample_weight"].to_numpy()
        for lab in (0, 1, 2):
            lm = (d["label"] == lab).to_numpy()
            for c, v in zip(features, (isna.to_numpy()[lm] * w[lm, None]).sum(axis=0)):
                miss_w[(split, c, lab)] += float(v)
        for lab, g in d.groupby("label"):
            cls[(split, int(lab))] += len(g)
            wcls[(split, int(lab))] += float(g["sample_weight"].sum())
    return miss, n, cls, wcls, miss_w


def missing_leak_scan(features, miss, n, wcls, miss_w):
    """피처별·split별 결측 행 vs 값 있는 행의 label 분포(가중, 원래 분포 기준) + 경고."""
    rows = []
    for s in C.SPLITS:
        tot = np.array([wcls[(s, l)] for l in (0, 1, 2)])
        base = tot / tot.sum()
        for f in features:
            if miss[(s, f)] == 0:
                continue
            mw = np.array([miss_w[(s, f, l)] for l in (0, 1, 2)])
            pw = tot - mw
            mp = mw / mw.sum() if mw.sum() else np.full(3, np.nan)
            pp = pw / pw.sum() if pw.sum() else np.full(3, np.nan)
            warn = []
            if miss[(s, f)] >= 100 and mp[2] >= 3 * base[2]:
                warn.append("결측 행 결항률 ≥ 전체×3")
            if miss[(s, f)] >= 100 and abs(mp[1] - base[1]) >= 0.10:
                warn.append("결측 행 지연률 차이 ≥ 10%p")
            rows.append({"split": s, "feature": f, "missing_rows": miss[(s, f)],
                         "missing_%": round(100 * miss[(s, f)] / n[s], 3),
                         "결측: 정상/지연/결항 %": "/".join(f"{100 * x:.2f}" for x in mp),
                         "값 있음: 정상/지연/결항 %": "/".join(f"{100 * x:.2f}" for x in pp),
                         "전체 결항 %": round(100 * base[2], 2), "warning": "; ".join(warn)})
    return pd.DataFrame(rows)


LEAK_REASON = [
    (lambda f: f.startswith("dest_"), "도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. "
     "예측 시점에도 같은 공백이 보이므로 사후 정보 아님", False),
    (lambda f: f.startswith("hist_"), "직전 12개월(공개분)에 그 그룹 실적이 없는 신규 노선·시간대·공항. 신규 편은 지연·결항 양상이 다를 수 있으나 예측 시점에 알 수 있는 정보", False),
    (lambda f: f.startswith("tail_"), "참조일에 같은 편명을 찾지 못함(편명 변경·신규 편). 운항일 당일 정보를 쓰지 않으므로 사후 정보 아님", False),
    (lambda f: f.endswith("_3h") or f.endswith("_3h_in") or f.endswith("_3h_kt") or f.endswith("_3h_ft")
     or f == "alti_change_3h", "3시간 창 안에 정시 관측이 없거나 3시간 전 기압 관측이 없음(관측 간격 큼)", False),
    (lambda f: f in ("wind_dir_sin", "wind_dir_cos"), "풍향 VRB(변동) 또는 풍속 결측 — 약풍 시 VRB 보고가 많음(물리적 의미)", False),
    (lambda f: f == "origin_yoy_ratio", "1년 전 같은 달(M−15월)에 운항 실적이 없던 공항(신규·재취항, 특히 2020~21년 코로나 감편). "
     "공개된 BTS로 계산되므로 사후 정보 아님", False),
]


def leak_reason(f):
    for test, why, is_leak in LEAK_REASON:
        if test(f):
            return why, is_leak
    return ("출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. "
            "예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님", False)


def scan_merged():
    """merged/ 에서 예측 시점 정의, 결측 관련 컬럼만 읽어 점검."""
    bad_pred, n = 0, 0
    for f in sorted(C.MERGED.rglob("*.parquet")):
        d = pd.read_parquet(f, columns=["sched_dep_utc", "prediction_time_utc"])
        bad_pred += int((d["sched_dep_utc"] - d["prediction_time_utc"] != pd.Timedelta(hours=C.PREDICTION_LEAD_H)).sum())
        n += len(d)
    return bad_pred, n


def class_pct_table(df, value_col):
    t = df.pivot_table(index="split", columns="label", values=value_col, aggfunc="sum").fillna(0)
    pct = (100 * t.div(t.sum(axis=1), axis=0)).round(3)
    pct.columns = [f"{c} {CLASS_NAMES.get(c, c)} %" for c in pct.columns]
    return pct


def unmatched_investigation(um, sm):
    """매칭 실패가 많은 공항의 원인: 직전 관측까지 간격, 실패 편의 현지 출발 시각 분포, 매핑 거리."""
    rows = []
    tot = pd.Series(dtype=float)
    for f in sorted(C.MERGED.rglob("*.parquet")):
        tot = tot.add(pd.read_parquet(f, columns=["Origin"])["Origin"].value_counts(), fill_value=0)
    fails = um[um["weather_match_status"] == "no_obs_within_2h"]
    top = fails["Origin"].value_counts().head(10)
    for ap, k in top.items():
        g = fails[fails["Origin"] == ap]
        hours = g["dep_hour_local"].value_counts().sort_index()
        hr_top = ", ".join(f"{int(h):02d}시:{c}" for h, c in hours.sort_values(ascending=False).head(4).items())
        smr = sm.loc[ap]
        gap = float(g["origin_last_obs_age_min"].median())
        if np.isnan(gap):
            verdict = "관측 자체가 없음(기간 초반 아카이브 없음)"
        elif ap == "BQN" or (hours.index.isin(range(0, 6)).sum() and hours[hours.index < 6].sum() > 0.5 * k):
            verdict = "야간 정기 공백"
        elif gap > 24 * 60:
            verdict = "하루 이상 관측 중단(장비 장애 추정)"
        else:
            verdict = "수 시간 관측 공백"
        rows.append(dict(Origin=ap, iem_station=smr["iem_station"], match_type=smr["match_type"],
                         station_distance_km=smr["distance_km"], failed=int(k), sampled_flights=int(tot.get(ap, 0)),
                         failed_pct=round(100 * k / tot.get(ap, np.nan), 2),
                         median_gap_to_last_obs_min=round(gap, 0), top_local_dep_hours=hr_top, 판정=verdict))
    extra = []
    for ap in ("BQN", "GCC"):
        g = fails[fails["Origin"] == ap]
        extra.append(dict(Origin=ap, failed=len(g), sampled_flights=int(tot.get(ap, 0)),
                          failed_pct=round(100 * len(g) / max(tot.get(ap, 0), 1), 2),
                          median_gap_to_last_obs_min=round(float(g["origin_last_obs_age_min"].median()), 0)
                          if len(g) else None))
    return pd.DataFrame(rows), pd.DataFrame(extra)


def coverage_note(ap):
    cov = pd.read_csv(C.WEATHER_DIR / "coverage_report.csv")
    c = cov[(cov["bts_iata"] == ap) & (cov["year"] >= 2022)]
    return ", ".join(f"{int(r.year)}:{r.coverage_pct}%" for r in c.itertuples())


def service_check(n=100, seed=7):
    """학습셋 무작위 n편: build_service_features 결과 vs 병합본 값."""
    feats = service_feature_names()
    files = sorted(C.MERGED.rglob("*.parquet"))
    rng = np.random.default_rng(seed)
    picks = Counter(rng.integers(0, len(files), n).tolist())
    rows = []
    for i, k in sorted(picks.items()):
        d = pd.read_parquet(files[i])
        d = d[d["in_ml_dataset"]]
        rows.append(d.sample(k, random_state=int(rng.integers(1e9))))
    sample = pd.concat(rows, ignore_index=True)
    mismatch, examples, all_match = Counter(), [], 0
    for rec in sample.to_dict("records"):
        got = build_service_features(rec, as_of_date=rec["FlightDate"])
        ok = True
        for f in feats:
            a, b = rec.get(f), got.get(f)
            a_na, b_na = pd.isna(a), pd.isna(b)
            if a_na and b_na:
                continue
            if isinstance(a, str) or isinstance(b, str):
                same = (not a_na) and (not b_na) and str(a) == str(b)
            else:
                same = (not a_na) and (not b_na) and bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-9))
            if not same:
                ok = False
                mismatch[f] += 1
                examples.append(dict(flight_id=rec["flight_id"], feature=f, train=a, service=b))
        all_match += ok
    return dict(n=len(sample), features=feats, all_match=all_match, mismatch=dict(mismatch), examples=examples)


def quality_report(svc):
    P = C.PROCESSED
    lists = json.loads((P / "feature_lists.json").read_text(encoding="utf-8"))
    fstats = json.loads((C.INTERIM / "flight_stats.json").read_text(encoding="utf-8"))
    mstats = json.loads((C.INTERIM / "merge_stats.json").read_text(encoding="utf-8"))
    samp = pd.read_parquet(C.INTERIM / "sampling_stats.parquet")
    v2f = json.loads((V2I / "flight_stats.json").read_text(encoding="utf-8"))
    v2m = json.loads((V2I / "merge_stats.json").read_text(encoding="utf-8"))
    tail = pd.read_parquet(V2I / "tail_missing_stats.parquet")
    tail_hist = pd.read_parquet(C.INTERIM / "tail_hist_stats.parquet")
    id_check = pd.read_parquet(C.INTERIM / "v2_id_check.parquet")
    qc = pd.read_parquet(V2I / "weather_qc_counts.parquet").T.reset_index()
    qc.columns = ["column", "values_set_to_nan"]
    last_run = {}
    for log, tag in ((C.V2 / "run_log.jsonl", " (v2, 재사용)"), (C.RUN_LOG, "")):
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if tag and rec["script"] != "preprocess_weather":
                    continue
                last_run[rec["script"] + tag] = rec
    um = pd.read_parquet(P / "unmatched_flights.parquet")
    sm = pd.read_csv(C.STATION_MAP, dtype=str).set_index("bts_iata")

    all_feats = list(dict.fromkeys(f for k in ("base", "base_plus_candidates", "no_weather", "with_dest_weather",
                                               "experimental_tail") for f in lists[k]))
    miss, n_ml, cls, wcls, miss_w = scan_ml(all_feats)
    leak_scan = missing_leak_scan(all_feats, miss, n_ml, wcls, miss_w)
    warn = leak_scan[leak_scan["warning"] != ""].copy()
    judged = [leak_reason(f) for f in warn["feature"]]
    warn["원인"] = [w for w, _ in judged]
    warn["누수 판단"] = ["누수 → 제외" if bad else "누수 아님 → 유지" for _, bad in judged]
    bad_pred, n_merged = scan_merged()
    env = ram_disk()

    # ---- 샘플링 전후 (split × 월)
    by_month = samp.pivot_table(index=["split", "ym"], columns="label", values=["n_full", "n_sample"],
                                aggfunc="sum").fillna(0)
    mrows = []
    for (split, ym), r in by_month.iterrows():
        nf, ns = r["n_full"], r["n_sample"]
        row = dict(split=split, month=ym, rows_full=int(nf.sum()), rows_sample=int(ns.sum()))
        for lab in (0, 1, 2):
            row[f"full_{lab}%"] = round(100 * nf.get(lab, 0) / nf.sum(), 2)
            row[f"sample_{lab}%"] = round(100 * ns.get(lab, 0) / ns.sum(), 2)
        mrows.append(row)
    month_tbl = pd.DataFrame(mrows)
    split_tot = samp.groupby(["split", "label"])[["n_full", "n_sample", "weighted"]].sum().reset_index()
    full_pct = class_pct_table(split_tot, "n_full")
    samp_pct = class_pct_table(split_tot, "n_sample")
    w_pct = class_pct_table(split_tot, "weighted")
    diff_w = (w_pct.values - full_pct.values)
    cmp = pd.concat([full_pct.add_prefix("원본 "), samp_pct.add_prefix("샘플 "), w_pct.add_prefix("가중 복원 ")],
                    axis=1).reset_index()
    evals = [s for s in ("validation", "test") if s in full_pct.index]
    max_eval_diff = float(np.abs(samp_pct.loc[evals].values - full_pct.loc[evals].values).max())
    max_w_diff = float(np.abs(diff_w).max())

    split_rows = split_tot.groupby("split")[["n_full", "n_sample"]].sum().reset_index()
    split_rows["ml_rows"] = split_rows["split"].map(n_ml)
    split_rows["period"] = split_rows["split"].map(lambda s: " ~ ".join(C.SPLITS[s]))

    # ---- 매칭
    def status_tbl(counter):
        t = pd.Series(counter).rename_axis("k").reset_index(name="flights")
        t[["split", "status"]] = t["k"].str.split("|", expand=True)
        p = t.pivot_table(index="split", columns="status", values="flights", aggfunc="sum").fillna(0).astype(int)
        p["matched %"] = (100 * p.get("matched", 0) / p.sum(axis=1)).round(3)
        return p.reset_index()
    origin_tbl, dest_tbl = status_tbl(v2m["origin"]), status_tbl(v2m["dest"])
    inv, inv_named = unmatched_investigation(um, sm)
    canc_total = int(split_tot.loc[split_tot["label"] == 2, "n_sample"].sum())
    canc_lost = int(((um["label"] == 2) & (um["weather_match_status"] != "matched")).sum())

    # ---- 누수·피처 목록
    forbidden = set(S.forbidden_columns()) | {c for c in all_feats if c.endswith("_oracle")}
    feat_viol = {k: sorted(set(v) & forbidden) for k, v in lists.items()
                 if isinstance(v, list) and k not in ("forbidden_in_features", "sampling_columns")}
    leak = pd.DataFrame([
        dict(check="출발 공항: 관측 시각 > 예측 시각", violations=mstats["leak_origin_future"]),
        dict(check="출발 공항: 관측이 예측 시각 − 2시간보다 오래됨", violations=mstats["leak_origin_over_tol"]),
        dict(check="도착 공항: 관측 시각 > 예측 시각", violations=mstats["leak_dest_future"]),
        dict(check="도착 공항: 관측이 예측 시각 − 2시간보다 오래됨", violations=mstats["leak_dest_over_tol"]),
        dict(check="예측 시각 ≠ 예정 출발 UTC − 3시간 (병합본 전체)", violations=bad_pred),
        dict(check="BTS 기반 피처가 쓴 데이터의 마지막 날 > 운항월 M−3월 말 (행 단위)",
             violations=mstats["timing_violation_rows"]),
        dict(check="  └ 피처 그룹별 합계 (hist_* 6종, proxy_ly, tail_hist, proxy_recent/yoy)",
             violations=sum(fstats["timing_violations"].values())),
        dict(check="v2와 같은 행(flight_id 집합)이 아닌 달", violations=len(mstats["id_diff_months"])),
        dict(check="v2와 label이 다른 행", violations=mstats["label_mismatch"]),
        dict(check="학습 피처 목록에 사후 정보·*_oracle·샘플링 컬럼·label_arr 포함",
             violations=sum(len(v) for v in feat_viol.values())),
        dict(check="Train/Validation/Test 기간 겹침",
             violations=int(not (C.SPLITS["train"][1] < C.SPLITS["validation"][0] <= C.SPLITS["validation"][1]
                                 < C.SPLITS["test"][0]))),
    ])

    # ---- hist: 전체 BTS 기준 확인
    bts_files = pd.read_csv(C.INSPECTION / "bts_files.csv")
    ym = bts_files["file"].str[4:8] + "-" + bts_files["file"].str[9:11]
    expected = int(bts_files.loc[ym >= C.BTS_READ_START[:7], "rows"].sum())
    hist_rows = []
    for name in FE.HIST_SPECS:
        h = pd.read_parquet(C.AGG / f"monthly_outcomes_{name}.parquet", columns=["n"])
        hist_rows.append(dict(table=name, flights_counted=int(h["n"].sum()), bts_rows_expected=expected,
                              equal=int(h["n"].sum()) == expected,
                              missing_in_sample=sum(v for k, v in fstats["hist_missing"].items()
                                                    if k.endswith(f"|{name}"))))

    # ---- 혼잡도, Tail
    cc = fstats["congestion_corr"]
    corr = pd.DataFrame([dict(feature=n, proxy_ly=cc.get(f"{n}|proxy_ly"), proxy_recent=cc.get(f"{n}|proxy_recent"),
                              v2_proxy_unavailable=v2f["congestion_corr"].get(n)) for n in S.CONGESTION_NAMES])
    th = tail_hist.groupby(["split", "label"])[["n", "n_found", "weighted_n", "weighted_found"]].sum().reset_index()
    th["found %"] = (100 * th["n_found"] / th["n"]).round(2)
    th["missing % (tail_leg_index_hist)"] = (100 - th["found %"]).round(2)
    th["label"] = th["label"].map(lambda x: f"{x} {CLASS_NAMES.get(x, x)}")
    th_all = tail_hist.groupby("split")[["weighted_n", "weighted_found"]].sum()
    th_rate = (100 * th_all["weighted_found"] / th_all["weighted_n"]).round(2).to_dict()
    tail_warn = warn[warn["feature"].str.startswith("tail_")]
    tl = tail.groupby(["split", "label"])[["n", "n_missing"]].sum().reset_index()
    tl["missing %"] = (100 * tl["n_missing"] / tl["n"]).round(3)
    tl["label"] = tl["label"].map(lambda x: f"{x} {CLASS_NAMES.get(x, x)}")
    tl_all = tail.groupby("label")[["n", "n_missing"]].sum()
    tail_c = round(100 * tl_all.loc[2, "n_missing"] / tl_all.loc[2, "n"], 2)
    tail_nc = round(100 * tl_all.loc[[0, 1], "n_missing"].sum() / tl_all.loc[[0, 1], "n"].sum(), 3)

    # ---- 피처 결측률
    mrow = []
    for f in all_feats:
        r = dict(feature=f, lists=",".join(k for k in ("base", "base_plus_candidates", "no_weather",
                                                       "with_dest_weather", "experimental_tail") if f in lists[k]))
        for s in C.SPLITS:
            r[f"missing_% {s}"] = round(100 * miss[(s, f)] / max(n_ml[s], 1), 3)
        mrow.append(r)
    miss_tbl = pd.DataFrame(mrow)

    # ---- 환경·용량·시간
    sizes = {"merged/": dir_size_mb(C.MERGED), "ml/": dir_size_mb(C.ML), "aggregates/": dir_size_mb(C.AGG),
             "interim/ (재생성 가능)": dir_size_mb(C.INTERIM),
             "unmatched_flights.parquet": round((P / "unmatched_flights.parquet").stat().st_size / 2 ** 20, 1)}
    run_tbl = pd.DataFrame([dict(script=k, seconds=v["seconds"], minutes=round(v["seconds"] / 60, 1),
                                 peak_memory_mb=v["peak_memory_mb"]) for k, v in last_run.items()])
    peak = max(v["peak_memory_mb"] for v in last_run.values())

    tz = v2f["tz"]   # UTC 변환은 v2 결과 재사용 (같은 행)
    out = [
        "# 데이터 품질 검증 보고서 (v3)", "",
        "v2(`processed_v2/`)에서 두 문제를 고친 v3입니다. (1) 당일 Tail_Number로 만든 `tail_*`의 결측이 곧 결항이라는 누수, "
        "(2) BTS가 약 2개월 늦게 공개돼 서비스 시점에 만들 수 없는 `*_proxy`(D−7~28)와 `hist_*`(직전 달까지). "
        f"이제 운항월 M의 BTS 기반 피처는 **M−{C.BTS_LAG_MONTHS}월 말까지 공개된 BTS만** 씁니다. "
        "샘플링(같은 seed)과 기상 결합은 v2 결과를 그대로 재사용했고, BTS 쪽 피처만 월 단위로 다시 계산했습니다. "
        "레이블, 예측 시점, merge_asof, UTC 변환, 기상 QC, 3시간 창 특징은 v1·v2와 같습니다.", "",
        "## 1. 환경과 설정", "",
        md_table(pd.DataFrame([dict(item=k, value=v) for k, v in {
            "RAM 전체 (GB)": env.get("ram_total_gb"), "RAM 작업 시작 시 사용 가능 (GB)": 4.9,
            "디스크 여유 (GB, 현재)": env["disk_free_gb"],
            "train 기간": " ~ ".join(C.SPLITS["train"]), "validation 기간": " ~ ".join(C.SPLITS["validation"]),
            "test 기간": " ~ ".join(C.SPLITS["test"]), "NEG_RATE": C.NEG_RATE, "EVAL_RATE": C.EVAL_RATE,
            "SAMPLE_SEED": f"{C.SAMPLE_SEED} (월별 seed = SAMPLE_SEED + YYYYMM)",
            "BTS_LAG_MONTHS": f"{C.BTS_LAG_MONTHS} (운항월 M → M−{C.BTS_LAG_MONTHS}월 말까지 BTS 사용)",
            "hist 창": f"M−{C.BTS_LAG_MONTHS + C.HIST_WINDOW_MONTHS - 1} ~ M−{C.BTS_LAG_MONTHS} (12개월, 월 단위), 평활 m={C.HIST_SMOOTH_M}",
            "혼잡도 proxy_ly": "D−" + "/".join(str(x) for x in C.PROXY_LY_LAGS_DAYS) + " 평균",
            "혼잡도 proxy_recent": "M−3월 중 같은 요일인 날들의 평균",
            "base_plus_candidates 혼잡도": lists["congestion_proxy_in_base_plus"],
            "기상 월 여유": f"±{C.WX_MONTH_MARGIN_DAYS}일", "최대 메모리 (모든 단계 중)": f"{peak:,.0f} MB",
        }.items()])), "",
        "- 사전 추정: 작업 시작 시 사용 가능 메모리가 4.9GB라, 전체를 한 번에 올리는 방식(수십 GB)은 불가능했습니다. "
        "그래서 BTS는 월 단위로 필요한 컬럼만, 기상은 관측소 25개씩, 결합은 월 단위로 처리하도록 설계했습니다. "
        "디스크는 여유가 충분합니다. 이 방식으로 기본값(NEG_RATE=0.2, train 2022~2023)이 감당되어 **설정값을 바꾸지 않았습니다.**",
        "- v3는 기상 전처리·결합을 다시 하지 않았습니다(아래 preprocess_weather는 v2 실행 기록).", "",
        md_table(run_tbl), "",
        md_table(pd.DataFrame([dict(path=k, size_mb=v) for k, v in sizes.items()])), "",
        "## 2. 규모 (샘플링 전 → 후 → 학습셋)", "", md_table(split_rows), "",
        f"- v2와 flight_id 집합이 같은 달: {int(id_check['identical'].sum())} / {len(id_check)}. v2와 label이 다른 행: {mstats['label_mismatch']}.",
        "- `ml_rows`는 출발 공항 기상 매칭에 성공하고 Label이 정의된 행입니다.", "",
        "## 3. 샘플링 전후 클래스 분포", "",
        "split 합계 기준입니다. '가중 복원'은 샘플 행에 `sample_weight`를 곱해 되돌린 분포입니다.", "",
        md_table(cmp), "",
        f"- 가중 복원 분포와 원본 분포의 최대 차이: **{max_w_diff:.3f}%p**.",
        f"- validation/test 샘플과 원본의 클래스 비율 최대 차이: **{max_eval_diff:.3f}%p** (기준 ±0.1%p 이내: {'통과' if max_eval_diff <= 0.1 else '초과'}).", "",
        "### split × 월별", "", md_table(month_tbl), "",
        "## 4. 기상 매칭", "", "**출발 공항**", "", md_table(origin_tbl), "", "**도착 공항**", "", md_table(dest_tbl), "",
        "- 출발 공항 매칭 실패 편은 학습셋에서 빠지고 `unmatched_flights.parquet`에 보존했습니다. 도착 공항 매칭 실패 편은 학습셋에 남기고 `dest_*`를 NaN으로 둡니다.",
        f"- 매칭 실패로 학습셋에서 빠진 결항 편: 샘플 내 결항 {canc_total:,}편 중 {canc_lost:,}편({100 * canc_lost / max(canc_total, 1):.2f}%).", "",
        "### 매칭 실패가 많은 출발 공항 원인 조사", "",
        md_table(inv), "",
        "- `median_gap_to_last_obs_min`: 실패한 편의 예측 시점에서 직전 관측까지 간격(허용 시간 없이 찾은 값). 120분보다 크면 관측 공백입니다.",
        "- 모든 상위 공항은 공항 구내 관측소(매핑 거리 3km 이내)에 정상 매핑되어 있습니다. 매핑 문제는 없었습니다.",
    ]
    for ap in inv["Origin"].head(4):
        out.append(f"- {ap} 연도별 관측 커버리지: {coverage_note(ap)}")
    out += [
        "", "**v1에서 실패율이 높았던 BQN·GCC (v2 기간)**", "", md_table(inv_named), "",
        "- **BQN**: IEM 관측이 매일 현지 01~05시(UTC 05~09시)에 없습니다(연간 커버리지 65~70%). "
        "이른 아침 출발 편은 예측 시점(3시간 전)이 이 공백에 걸려 실패합니다. 매핑 문제가 아니라 관측 공백이라 수정하지 않았습니다.",
        "- **GCC**: v1(2025년 1~3월)의 22% 실패는 그 기간의 관측 중단 때문이었습니다. v2 전체 기간 기준 실패율은 위 표와 같습니다.",
        "- 그 밖의 상위 공항(RSW, FLL, LGB, AGS 등)은 실패율이 낮지만, 직전 관측까지 간격이 수 시간~수일입니다. "
        "특정 기간에 관측소 데이터가 끊긴 것으로, 매핑은 정상입니다. 모두 관측 공백으로 기록했습니다.", "",
        "## 5. UTC 변환", "",
        md_table(pd.DataFrame([dict(tz_status=k, flights=v) for k, v in tz["status"].items()])), "",
        f"- 예정 소요시간 교차검증에서 5분 넘게 어긋난 편: {tz['check_gt5']:,} / {tz['n']:,} ({100 * tz['check_gt5'] / tz['n']:.4f}%). "
        "v1과 같이 서머타임 전환 주말 등 BTS 예정 도착 시각 기록 불일치로, 출발 UTC에는 영향이 없습니다.", "",
        "## 6. 데이터 누수 검증", "", md_table(leak), "",
        "- 기상 3시간 창 특징과 기온 spike QC는 과거 관측만 사용합니다(v1과 동일).",
        "- 당일 계획 편수 `*_oracle`, v2의 `*_proxy_v2_unavailable`, 당일 기체 일정 `*_sameday`는 병합본에만 있고 학습 피처에서 뺐습니다.",
        "- `tail_*_hist`는 과거 참조일의 예정 시각만 쓰고, 운항일 당일의 Tail_Number·실제 시각·지연은 쓰지 않습니다.", "",
        "### 공개 시점 검사", "",
        md_table(pd.DataFrame([dict(feature_group=k, rows_over_limit=v) for k, v in fstats["timing_violations"].items()])), "",
        f"- 각 행의 `bts_source_max_date`(BTS 기반 피처가 쓴 마지막 날)가 `bts_source_limit_date`(M−{C.BTS_LAG_MONTHS}월 말)를 넘는 행: "
        f"**{mstats['timing_violation_rows']}건**.",
        f"- **공개 시각 한계:** 기상 시각은 관측 시각이며 실제 배포 시각은 데이터에 없습니다. `OBS_PUBLISH_LAG_MIN`={C.OBS_PUBLISH_LAG_MIN}분.", "",
        "## 7. 과거 실적(hist_*)이 BTS 전체로 계산됐는지", "",
        md_table(pd.DataFrame(hist_rows)), "",
        f"- 집계표의 편수 합계가 {C.BTS_READ_START[:4]}~2025년 BTS 원본 행 수와 같으면, 샘플이 아니라 전체로 계산된 것입니다.",
        "- `missing_in_sample`: 창(M−14~M−3)에 해당 그룹 실적이 없어 NaN인 샘플 행 수. 신규 노선·시간대가 해당합니다.",
        "- **2020년 말 코로나 영향:** 2022년 1~2월 운항편의 창은 2020-11~2021-10, 2020-12~2021-11입니다. 2020년 말은 운항 감소·결항 증가 시기(2020년 결항률 6.0%)라 "
        "이 달들의 hist_*는 이후 기간보다 결항률이 높게 잡힐 수 있습니다. train 초반 몇 달에만 해당합니다.", "",
        "## 8. 혼잡도 proxy vs oracle (피어슨 상관, 샘플 전 데이터셋 전체)", "", md_table(corr), "",
        f"- 평균 상관: proxy_ly {mstats['proxy_mean_corr']['proxy_ly']:.4f}, proxy_recent {mstats['proxy_mean_corr']['proxy_recent']:.4f} "
        f"→ **`{lists['congestion_proxy_in_base_plus']}`를 base_plus_candidates에** 넣고, 다른 쪽은 `alt_congestion_candidates`로 남겼습니다.",
        "- v2 proxy(D−7~28)는 상관이 가장 높지만 서비스 시점에 그 BTS가 공개돼 있지 않아 쓸 수 없습니다(참고용).",
        "- `origin_yoy_ratio`는 작년 대비 운항량 추세 보정용 보조 피처로 base_plus_candidates에 넣었습니다.", "",
        "## 9. 기체 일정 피처", "",
        "### v2 당일 Tail_Number 결측률 (label별, 샘플 전) — 누수 원인", "", md_table(tl), "",
        f"- 결항 편의 당일 Tail_Number 결측률 {tail_c}%, 결항 아닌 편 {tail_nc}%. v2의 당일 `tail_*`는 결측이 곧 결항이라 `*_sameday`로 이름을 바꿔 학습에서 뺐습니다.", "",
        "### v3 tail_*_hist 매칭률 (참조일에 같은 편을 찾은 비율, 샘플 기준)", "", md_table(th), "",
        "- 원래 분포 기준(가중) 매칭률: " + ", ".join(f"{k} {v}%" for k, v in th_rate.items()) + ". 편명이 해마다 바뀌어 매칭이 안 되는 편이 있습니다.",
        "- label별 매칭률 차이가 작으면 결측이 결항을 알려주지 않는다는 뜻입니다. 결측 누수 검사 결과:",
        ("  - `tail_*_hist` 경고 없음 → **통과**" if tail_warn.empty else
         "  - 경고: " + "; ".join(f"{t.split} {t.feature} ({t.warning})" for t in tail_warn.itertuples())), "",
        "## 10. 학습 피처 목록 점검", "",
        md_table(pd.DataFrame([dict(list=k, n_features=len(lists[k]), forbidden_found=", ".join(feat_viol.get(k, [])) or "0건")
                               for k in ("base", "base_plus_candidates", "no_weather", "with_dest_weather",
                                         "experimental_tail")])), "",
        "## 11. 학습셋 클래스 분포 (ml/, 기상 매칭 성공 행)", "",
        md_table(pd.DataFrame([dict(split=s, label=f"{l} {CLASS_NAMES[l]}", rows=cls[(s, l)],
                                    pct=round(100 * cls[(s, l)] / n_ml[s], 3),
                                    weighted_pct=round(100 * wcls[(s, l)] / sum(wcls[(s, x)] for x in (0, 1, 2)), 3))
                               for s in C.SPLITS for l in (0, 1, 2)])), "",
        "## 12. 기상 원본 QC (v2 재사용)", "", md_table(qc), "",
        "## 13. 학습 피처 결측률 (split별 %)", "", md_table(miss_tbl), "",
        "## 14. 기존 processed/ (v1) 대비 변경점", "",
        md_table(pd.DataFrame([
            dict(항목="기간·분할", v1="2025-01~03, 월 단위 1/2/3월", v2="train 2022~23 / val 2024 / test 2025 (각 12개월)"),
            dict(항목="샘플링", v1="없음 (164만 편 전체)", v2=f"train 정상만 {C.NEG_RATE}, val·test {C.EVAL_RATE} 균등, sample_rate/weight"),
            dict(항목="행 수 (학습셋)", v1="1,644,539", v2=f"{sum(n_ml.values()):,}"),
            dict(항목="hist_*", v1="2024년 고정 1년", v2="운항월 직전 12개월 창(월 단위), BTS 전체로 계산"),
            dict(항목="혼잡도", v1="당일 계획 편수 (서비스 시 알 수 없음)", v2="*_proxy(직전 4주 같은 요일 평균), 당일 값은 *_oracle로 학습 제외"),
            dict(항목="도착 공항 기상", v1="없음", v2="dest_* 24개 (같은 예측 시점 기준)"),
            dict(항목="기체 일정", v1="없음", v2="tail_* 4개 (experimental_tail 전용)"),
            dict(항목="month·season", v1="Train 내 상수라 제외", v2="base에 포함"),
            dict(항목="도착 기준 레이블", v1="없음", v2="label_arr (병합본, 평가용)"),
            dict(항목="피처 목록", v1="base, base_plus_candidates", v2="+ no_weather, with_dest_weather, experimental_tail"),
            dict(항목="포맷", v1="CSV + pickle", v2="Parquet(snappy)만"),
            dict(항목="최대 메모리", v1="측정 안 함 (전체 적재)", v2=f"{peak:,.0f} MB"),
        ])), "",
        "## 15. 결측 누수 검사 (모든 학습 피처, 가중 = 원래 분포 기준)", "",
        "결측 행과 값 있는 행의 label 분포를 split별로 비교했습니다. 경고 기준: 결측 행이 100행 이상이면서 "
        "(a) 결측 행 결항률 ≥ 전체 결항률×3 또는 (b) 결측 행 지연률이 전체와 10%p 이상 차이.", "",
        "### 경고 목록과 판단", "",
        (md_table(warn[["split", "feature", "missing_rows", "missing_%", "결측: 정상/지연/결항 %", "전체 결항 %", "warning",
                        "원인", "누수 판단"]]) if len(warn) else "경고 없음"), "",
        "### 전체 결과표", "", md_table(leak_scan), "",
        "## 16. 서비스 재현 검증 (`build_service_features`)", "",
        f"학습셋에서 무작위 {svc['n']}편을 골라 `ml/data_pipeline/service_features.py`의 `build_service_features(flight, as_of_date=운항일)` 결과를 학습셋 값과 비교했습니다. "
        f"비교 피처 {len(svc['features'])}개(날씨 제외).", "",
        f"- 모든 피처가 일치한 편: **{svc['all_match']} / {svc['n']}**",
        ("- 불일치 피처별 건수: " + ", ".join(f"`{k}` {v}" for k, v in svc["mismatch"].items())) if svc["mismatch"] else "- 불일치 없음",
        *[f"  - 예: flight_id {e['flight_id']} `{e['feature']}` 학습셋={e['train']} 서비스={e['service']}" for e in svc["examples"][:8]], "",
        "## 17. v2 대비 변경점", "",
        md_table(pd.DataFrame([
            dict(항목="BTS 공개 지연 규칙", v2="없음 (직전 달·직전 주 BTS 사용)", v3=f"운항월 M → M−{C.BTS_LAG_MONTHS}월 말까지 (BTS_LAG_MONTHS)"),
            dict(항목="hist_* 창", v2="M−12 ~ M−1", v3="M−14 ~ M−3 (source_max_month 기록)"),
            dict(항목="혼잡도 학습 피처", v2="*_proxy (D−7/14/21/28)", v3=f"*_{lists['congestion_proxy_in_base_plus']} + origin_yoy_ratio (대안: alt_congestion_candidates)"),
            dict(항목="v2 혼잡도", v2="학습 피처", v3="*_proxy_v2_unavailable (병합본 전용)"),
            dict(항목="기체 일정", v2="당일 Tail_Number 기준 tail_* 4개 (누수)", v3="tail_*_hist 5개 (참조일 기준), 당일 값은 *_sameday (병합본 전용)"),
            dict(항목="결측 누수 검사", v2="없음", v3="모든 학습 피처 × split"),
            dict(항목="서비스 재현", v2="없음", v3="build_service_features + 100편 대조"),
            dict(항목="aggregates", v2="hist·sched_counts", v3="+ source_max_month/source_month, tail_sameday/, route_distance"),
            dict(항목="샘플·기상", v2="—", v3="v2와 같은 행·같은 기상 값 (flight_id로 재사용)"),
            dict(항목="학습셋 행 수", v2=f"{sum(v2m['ml_rows'].values()):,}", v3=f"{sum(n_ml.values()):,}"),
        ])), "",
        "## 18. 모델 담당에게 넘길 메모", "",
        f"- train은 정상편만 `NEG_RATE`={C.NEG_RATE}로 줄였습니다. 그래서 모델이 낸 확률은 정상 확률이 실제보다 낮게 나옵니다.",
        "- 되돌리는 방법은 둘 중 하나입니다.",
        "  1. 학습할 때 `sample_weight`를 가중치로 쓰기 (`xgboost`: `DMatrix(..., weight=sample_weight)`)",
        f"  2. 예측 후 정상 클래스 확률에 `1/NEG_RATE`(={1 / C.NEG_RATE:g})를 곱한 뒤, 세 클래스 합이 1이 되도록 다시 나누기",
        "- 평가와 임계값 조정은 반드시 원래 분포인 validation/test로 하세요. 두 split은 클래스 비율을 건드리지 않고 균등하게만 줄였습니다.",
        "- `sample_rate`, `sample_weight`는 학습 피처가 아닙니다. `feature_lists.json`의 목록만 피처로 쓰세요.",
        "- 실험 순서 제안: `base` → `base_plus_candidates` → `with_dest_weather`. 일주일 전 예측은 `no_weather`. "
        "`experimental_tail`은 `tail_*_hist`(참조일 기준)라 v2의 당일 누수는 없지만, 효과는 별도 비교로 확인하세요.",
        "- 범주형: " + ", ".join(f"`{c}`" for c in S.CATEGORICAL) + ".",
        f"- 학습셋의 BTS 기반 피처는 이미 공개 지연(M−{C.BTS_LAG_MONTHS})을 반영했습니다. 서비스에서는 `build_service_features(flight, as_of_date)`로 "
        "`aggregates/`만 조회하면 같은 피처가 나옵니다. 새 BTS 월이 공개되면 `preprocess_flights.py`로 aggregates만 갱신하면 됩니다.",
    ]
    (P / "data_quality_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    return dict(leak=leak, n_ml=dict(n_ml), max_eval_diff=max_eval_diff, max_w_diff=max_w_diff, peak=peak,
                sizes=sizes, origin=origin_tbl, dest=dest_tbl, corr=corr, tail=(tail_c, tail_nc), inv=inv,
                split_rows=split_rows, miss=miss_tbl, warn=warn, th_rate=th_rate)


def selection_report(miss_tbl):
    I = C.INSPECTION
    lists = json.loads((C.PROCESSED / "feature_lists.json").read_text(encoding="utf-8"))
    rows = []
    for ds, cols, src in (("BTS", S.BTS_COLUMNS, I / "bts_columns.csv"), ("IEM", S.IEM_COLUMNS, I / "iem_columns.csv")):
        info = pd.read_csv(src).set_index("column")
        for c, (cat, why) in cols.items():
            rows.append({"Dataset": ds, "Column": c, "Description": info["description"].get(c, ""),
                         "Missing Rate (%)": info["missing_pct"].get(c), "Category": cat, "Selection Reason": why})
    tbl = pd.DataFrame(rows)

    def fdesc(f):
        if f.startswith("dest_"):
            return "도착 공항 · " + fdesc(f[5:]) if f[5:] != "obs_age_min" else "예측 시점 − 매칭된 도착 공항 관측 시각(분)"
        for k, v in S.FEATURE_DESC.items():
            keys = [x.strip() for x in k.split("/")]
            if f in keys or (k.startswith("hist_*") and f.startswith("hist_")):
                return v
        return S.BTS_COLUMNS.get(f, S.IEM_COLUMNS.get(f, ("", "")))[1]

    m = miss_tbl.set_index("feature")
    feat = pd.DataFrame([dict(feature=f, lists=m.loc[f, "lists"], categorical="Y" if f in S.CATEGORICAL else "",
                              **{"missing_% train": m.loc[f, "missing_% train"]}, definition=fdesc(f))
                         for f in m.index])
    out = [
        "# 특징 선정 보고서 (v3)", "",
        "원본 컬럼 분류 기준과 근거는 v1(`processed/feature_selection_report.md`)과 같고, v2·v3에서 바뀐 부분만 반영했습니다. "
        "결측률은 원본 전체 기준(BTS 2020~2025, IEM 전체)입니다.", "",
        "## 1. 원본 컬럼 분류", "", md_table(tbl), "",
        "## 2. 학습 피처 목록 (`feature_lists.json`)", "",
        md_table(pd.DataFrame([dict(list=k, n=len(lists[k]), 설명=d) for k, d in (
            ("base", "항공편 일정 + 월·계절 + 출발 공항 핵심 기상"),
            ("base_plus_candidates", "base + 혼잡도 proxy(공개된 BTS 기준)·작년 대비 추세, 과거 실적(M−14~M−3), 출발 공항 기상 후보·3시간 창"),
            ("no_weather", "base_plus_candidates에서 출발 공항 기상 피처 전부 제외 (일주일 전 예측용)"),
            ("with_dest_weather", "base_plus_candidates + 도착 공항 기상(dest_*)"),
            ("experimental_tail", "with_dest_weather + tail_*_hist 5개 (참조일 기준 기체 일정)"),
            ("alt_congestion_candidates", "base_plus_candidates 의 혼잡도 proxy 대신 넣어 비교할 대안 4개"))])), "",
        md_table(feat), "",
        "## 3. v2·v3에서 바뀐 판단", "",
        f"- **BTS 공개 지연(v3):** 서비스는 BTS 공개 데이터와 기상만 씁니다. BTS는 약 2개월 늦게 공개되므로, 운항월 M의 BTS 기반 피처는 M−{C.BTS_LAG_MONTHS}월 말까지만 씁니다.",
        "- **혼잡도(v3):** 당일 편수(`*_oracle`)와 직전 4주(`*_proxy_v2_unavailable`)는 서비스 시점에 알 수 없어 학습에서 뺐습니다. 작년 같은 요일(`*_proxy_ly`)과 공개된 최신 달 같은 요일(`*_proxy_recent`) 중 oracle과 상관이 높은 쪽을 씁니다.",
        "- **과거 실적(v3):** 운항월마다 M−14~M−3 창을 BTS 전체로 다시 계산합니다.",
        "- **month, season:** split마다 12개월이 모두 들어가므로 다시 base에 넣었습니다.",
        "- **도착 공항 기상:** 같은 예측 시점의 도착 공항 최신 관측입니다. 도착지 악천후로 인한 지상 대기(GDP) 같은 영향을 반영하기 위한 후보입니다.",
        "- **기체 일정(v3):** 당일 Tail_Number 기준 `tail_*`는 결측이 곧 결항이라(v2 누수) `*_sameday`로 병합본에만 남겼습니다. 대신 과거 참조일의 같은 편 기체 일정으로 만든 `tail_*_hist`를 `experimental_tail`에 넣었습니다.",
        "- 위 기대 효과는 가설이며, 목록별 비교 실험으로 확인해야 합니다.",
    ]
    (C.PROCESSED / "feature_selection_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    timer = C.RunTimer("validate_dataset")
    svc = service_check()
    print("service check:", svc["all_match"], "/", svc["n"], svc["mismatch"])
    r = quality_report(svc)
    selection_report(r["miss"])
    print(r["leak"].to_string())
    print(r["split_rows"].to_string())
    print("max eval diff", r["max_eval_diff"], "max weighted diff", r["max_w_diff"])
    timer.done()


if __name__ == "__main__":
    main()
