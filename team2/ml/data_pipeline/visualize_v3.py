"""processed_v3 발표용 시각화. 그림을 화면에 띄운다 (파일로 저장하지 않음).

비율은 모두 sample_weight를 곱한 추정치(= 샘플링 전 원래 분포)로 계산한다.
train은 정상편을 20%만 뽑았으므로 가중치 없이 세면 지연·결항 비율이 부풀려진다.

실행: python -X utf8 ml/data_pipeline/visualize_v3.py            (전체)
      python -X utf8 ml/data_pipeline/visualize_v3.py --only 3 4  (번호 지정)
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

try:  # 주피터에서는 그림을 셀 아래에 바로 표시 (이전에 Agg 백엔드가 설정돼 있어도 되돌린다)
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

def _find_root():
    """processed_v3/ 가 있는 데이터 폴더. 환경변수 AIRLINE_DATA_ROOT > 이 파일의 상위 폴더 > 현재 폴더의 상위 폴더.
    (주피터 셀에 붙여넣으면 __file__이 없으므로 현재 폴더부터 찾는다)"""
    if os.environ.get("AIRLINE_DATA_ROOT"):
        return Path(os.environ["AIRLINE_DATA_ROOT"]).resolve()
    try:
        start = Path(__file__).resolve()
    except NameError:
        start = Path.cwd()
    for d in [start, *start.parents, Path.cwd(), *Path.cwd().parents]:
        if (d / "processed_v3").is_dir():
            return d
    raise FileNotFoundError("processed_v3 폴더를 찾을 수 없음 — 환경변수 AIRLINE_DATA_ROOT 를 지정하세요")


ROOT = _find_root()
P = ROOT / "processed_v3"
PERIOD = ("2022-01", "2025-12")

# 색: 레이블별로 고정 (정상=회색, 지연=파랑, 결항=주황). 텍스트는 잉크 색만 사용.
C_NORMAL, C_DELAY, C_CANCEL = "#b8b7b1", "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
LABEL_NAME = {0: "정상", 1: "지연(15분+)", 2: "결항"}
SPLIT_NAME = {"train": "학습 2022–23", "validation": "검증 2024", "test": "테스트 2025"}
FCAT = ["VFR", "MVFR", "IFR", "LIFR"]
CARRIER = {"AA": "American", "AS": "Alaska", "B6": "JetBlue", "DL": "Delta", "F9": "Frontier", "G4": "Allegiant",
           "HA": "Hawaiian", "MQ": "Envoy", "NK": "Spirit", "OH": "PSA", "OO": "SkyWest", "UA": "United",
           "WN": "Southwest", "YX": "Republic", "9E": "Endeavor", "QX": "Horizon", "ZW": "Air Wisconsin",
           "YV": "Mesa", "C5": "CommutAir", "G7": "GoJet", "PT": "Piedmont", "EV": "ExpressJet"}

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 100, "savefig.dpi": 200,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10.5, "legend.frameon": False,
})


# ---------------------------------------------------------------- 데이터 로드
_cache = {}


def load(kind, cols):
    """ml/ 또는 merged/ 의 전 split을 필요한 컬럼만 읽는다 (+split, label, sample_weight)."""
    cols = list(dict.fromkeys(list(cols) + ["label", "sample_weight"]))
    key = (kind, tuple(cols))
    if key not in _cache:
        parts = []
        for split in SPLIT_NAME:
            # merged/는 year=/month= 하위 폴더라 폴더째 읽으면 파티션 컬럼이 충돌 → 파일별로 읽는다
            files = sorted((P / kind / f"split={split}").rglob("*.parquet"))
            d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
            d["split"] = split
            parts.append(d)
        _cache[key] = pd.concat(parts, ignore_index=True)
    return _cache[key]


def rates(df, by):
    """가중 지연·결항 비율. delay = label 1 비율, cancel = label 2 비율 (분모는 전체)."""
    w = df["sample_weight"]
    t = pd.DataFrame({"w": w, "d": w * (df["label"] == 1), "c": w * (df["label"] == 2)})
    for k in by if isinstance(by, list) else [by]:
        t[k] = df[k].values
    g = t.groupby(by, observed=True)[["w", "d", "c"]].sum()
    g["delay"], g["cancel"], g["share"] = g.d / g.w, g.c / g.w, g.w / g.w.sum()
    return g


def outcomes(name):
    """BTS 전체 월별 실적 집계(샘플링 전) — 기간 제한."""
    d = pd.read_parquet(P / "aggregates" / f"monthly_outcomes_{name}.parquet")
    return d[(d.ym >= PERIOD[0]) & (d.ym <= PERIOD[1])]


# ---------------------------------------------------------------- 공통 도우미
def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def finish(fig, name, title, subtitle=None):
    h = fig.get_figheight()  # 제목 영역을 인치 단위로 고정해 그림 높이와 무관하게 겹치지 않게 한다
    fig.suptitle(title, x=0.012, y=1 - 0.18 / h, ha="left", va="top", fontsize=16, fontweight="bold")
    if subtitle:
        fig.text(0.012, 1 - 0.55 / h, subtitle, ha="left", va="top", fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.95 if subtitle else 0.6) / h))
    plt.show()


def bar_labels(ax, bars, values, fmt=pct, horizontal=False):
    for b, v in zip(bars, values):
        if horizontal:
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt(v), va="center", fontsize=9.5,
                    color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(v), ha="center", va="bottom", fontsize=9.5,
                    color=INK)


def pct_axis(ax, axis="y"):
    f = matplotlib.ticker.PercentFormatter(1.0, decimals=None)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


# ---------------------------------------------------------------- 1. 클래스 분포
def fig_class_distribution():
    s = pd.read_parquet(P / "interim" / "sampling_stats.parquet")
    rows = []
    for split in SPLIT_NAME:
        g = s[s.split == split].groupby("label")[["n_full", "n_sample"]].sum()
        rows.append((SPLIT_NAME[split] + " · 원본", g.n_full))
        if split == "train":
            rows.append(("학습 2022–23 · 샘플링 후", g.n_sample))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.grid(axis="y", visible=False)
    for i, (name, n) in enumerate(rows[::-1]):
        share = n / n.sum()
        left = 0
        for lab, col in ((0, C_NORMAL), (1, C_DELAY), (2, C_CANCEL)):
            v = share.get(lab, 0)
            ax.barh(i, v, left=left, color=col, height=0.6, edgecolor=SURFACE, linewidth=2,
                    label=LABEL_NAME[lab] if i == 0 else None)
            if v > 0.04:
                ax.text(left + v / 2, i, pct(v), ha="center", va="center", fontsize=10,
                        color="white" if lab else INK)
            elif v > 0:
                ax.text(1.005, i, f"결항 {pct(v)}", va="center", fontsize=9.5, color=INK2)
            left += v
        ax.text(-0.005, i - 0.42, f"{int(n.sum()):,}편", ha="right", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlim(0, 1)
    pct_axis(ax, "x")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    finish(fig, "01_class_distribution", "레이블 분포: 결항은 1~2%뿐인 심한 불균형",
           "학습셋은 정상편을 20%만 남겨 지연·결항 비중을 키웠다 — 학습 시 sample_weight로 원래 분포를 복원")


# ---------------------------------------------------------------- 2. 월별 추이
def fig_monthly_trend():
    s = pd.read_parquet(P / "interim" / "sampling_stats.parquet")
    m = s.pivot_table(index="ym", columns="label", values="n_full", aggfunc="sum")
    tot = m.sum(axis=1)
    x = pd.to_datetime(m.index)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
    for ax, lab, col, title in ((axes[0], 1, C_DELAY, "지연(15분+) 비율"), (axes[1], 2, C_CANCEL, "결항 비율")):
        r = m[lab] / tot
        ax.plot(x, r, color=col, lw=2, marker="o", ms=4)
        ax.set_title(title, fontsize=12)
        pct_axis(ax)
        for k in r.nlargest(2).index:
            ax.annotate(f"{k} {pct(r[k])}", (pd.Timestamp(k), r[k]), xytext=(6, 4), textcoords="offset points",
                        fontsize=9, color=INK)
        for start, name in (("2022-01", "학습"), ("2024-01", "검증"), ("2025-01", "테스트")):
            ax.axvline(pd.Timestamp(start), color=INK2, lw=0.8, ls=":")
            if ax is axes[0]:
                ax.text(pd.Timestamp(start), ax.get_ylim()[1], f" {name}", va="top", fontsize=9, color=INK2)
        ax.set_ylim(bottom=0)
    finish(fig, "02_monthly_trend", "월별 지연·결항 비율 (2022–2025, BTS 전체 운항편)",
           "여름 성수기에 지연, 겨울 폭풍 시기에 결항이 뛴다 — 계절성이 뚜렷")


# ---------------------------------------------------------------- 3. 비행 기상 등급
def fig_flight_category():
    df = load("ml", ["flight_category"]).dropna(subset=["flight_category"])
    g = rates(df, "flight_category").reindex([0.0, 1.0, 2.0, 3.0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, col, title in ((axes[0], "delay", C_DELAY, "지연(15분+) 비율"), (axes[1], "cancel", C_CANCEL, "결항 비율")):
        b = ax.bar(FCAT, g[key], color=col, width=0.6)
        bar_labels(ax, b, g[key])
        ax.axhline((g[key] * g.w).sum() / g.w.sum(), color=INK2, lw=1, ls="--")
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", visible=False)
        pct_axis(ax)
        ax.set_xticks(range(4), [f"{c}\n운항 {pct(s)}" for c, s in zip(FCAT, g.share)])
    ratio = g.cancel.iloc[3] / g.cancel.iloc[0]
    finish(fig, "03_flight_category", f"출발 공항 기상이 나쁠수록 결항이 급증 (LIFR은 VFR의 {ratio:.1f}배)",
           "예측 시점(출발 3시간 전) 출발 공항 비행 기상 등급 · 점선 = 전체 평균 · 가중 추정 2022–2025")


# ---------------------------------------------------------------- 4. 기상 현상별
WX = {"wx_thunder": "뇌우", "wx_snow": "눈", "wx_freezing": "어는 비·안개", "wx_fog": "안개", "wx_rain": "비",
      "wx_hail": "우박", "wx_heavy": "강한 강수", "wx_mist_haze": "박무·연무", "strong_wind": "강풍",
      "low_visibility": "저시정", "low_ceiling": "낮은 운고"}


def fig_weather_phenomena():
    df = load("ml", list(WX) + ["weather_severity"])
    rows = []
    for c, name in WX.items():
        g = rates(df[df[c] == 1].assign(k=1), "k")
        rows.append((name, g.w.iloc[0], g.delay.iloc[0], g.cancel.iloc[0]))
    calm = df[(df[list(WX)] == 0).all(axis=1)].assign(k=1)
    gc = rates(calm, "k")
    t = pd.DataFrame(rows, columns=["name", "w", "delay", "cancel"]).sort_values("cancel")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, key, col, title in ((axes[0], "cancel", C_CANCEL, "결항 비율"), (axes[1], "delay", C_DELAY, "지연(15분+) 비율")):
        b = ax.barh(t.name, t[key], color=col, height=0.6)
        bar_labels(ax, b, t[key], horizontal=True)
        base = gc[key].iloc[0]
        ax.axvline(base, color=INK2, lw=1, ls="--")
        ax.text(base, len(t) - 0.4, f" 기상현상 없음 {pct(base)}", fontsize=9, color=INK2, va="bottom")
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, t[key].max() * 1.25)
        pct_axis(ax, "x")
    axes[0].set_yticks(range(len(t)), [f"{n}  ({w / 1e3:,.0f}천 편)" for n, w in zip(t.name, t.w)])
    finish(fig, "04_weather_phenomena", "어떤 기상 현상이 결항을 부르나",
           "출발 공항 최신 관측에서 해당 현상이 보고된 편 · 괄호 = 가중 운항편 수 · 점선 = 아무 현상도 없는 편")


# ---------------------------------------------------------------- 5. 출발×도착 기상 히트맵
def fig_origin_dest_heatmap():
    df = load("ml", ["flight_category", "dest_flight_category"]).dropna(subset=["flight_category",
                                                                                 "dest_flight_category"])
    g = rates(df, ["flight_category", "dest_flight_category"])["cancel"].unstack()
    g = g.reindex(index=[0.0, 1.0, 2.0, 3.0], columns=[0.0, 1.0, 2.0, 3.0])
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    ax.grid(False)
    im = ax.imshow(g.values, cmap=BLUES, vmin=0)
    for i in range(4):
        for j in range(4):
            v = g.values[i, j]
            ax.text(j, i, pct(v), ha="center", va="center", fontsize=11,
                    color="white" if v > np.nanmax(g.values) * 0.55 else INK)
    ax.set_xticks(range(4), FCAT)
    ax.set_yticks(range(4), FCAT)
    ax.set_xlabel("도착 공항 기상 등급")
    ax.set_ylabel("출발 공항 기상 등급")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, format=matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    cb.outline.set_visible(False)
    finish(fig, "05_origin_dest_heatmap", "도착 공항 날씨도 결항을 좌우한다",
           "출발 공항이 맑아도(VFR 행) 도착 공항이 LIFR이면 결항률이 크게 오른다 · 칸 = 결항 비율")


# ---------------------------------------------------------------- 6. 시간대별
def fig_hourly():
    df = load("ml", ["dep_hour_local"])
    g = rates(df, "dep_hour_local")
    g = g[g.share > 0.002]  # 운항이 거의 없는 새벽 시간 제외
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, col, title in ((axes[0], "delay", C_DELAY, "지연(15분+) 비율"), (axes[1], "cancel", C_CANCEL, "결항 비율")):
        ax.plot(g.index, g[key], color=col, lw=2, marker="o", ms=5)
        k = g[key].idxmax()
        ax.annotate(f"{int(k)}시 {pct(g[key][k])}", (k, g[key][k]), xytext=(-10, 8), textcoords="offset points",
                    ha="right", fontsize=9.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("예정 출발 시각 (현지)")
        ax.set_xticks(range(0, 24, 3))
        ax.set_ylim(bottom=0)
        pct_axis(ax)
    finish(fig, "06_hourly", "오후·저녁으로 갈수록 지연이 누적된다",
           "예정 출발 현지 시각별 · 운항 비중 0.2% 미만 시간대 제외 · 가중 추정 2022–2025")


# ---------------------------------------------------------------- 7. 공항 지도
def fig_airport_map():
    o = outcomes("hist_origin").groupby("Origin")[["n", "canc", "flown", "dly"]].sum()
    sm = pd.read_csv(ROOT / "weather" / "station_map.csv").set_index("bts_iata")[["lat", "lon"]]
    o = o.join(sm, how="inner")
    o = o[(o.lon > -126) & (o.lon < -66) & (o.lat > 24) & (o.lat < 50) & (o.n >= 5000)]  # 미국 본토
    o["cancel"] = o.canc / o.n
    o["delay"] = o.dly / o.flown
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, key, title in ((axes[0], "cancel", "결항 비율"), (axes[1], "delay", "지연(15분+) 비율")):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        d = o.sort_values("n")
        sc = ax.scatter(d.lon, d.lat, s=np.sqrt(d.n) / 4, c=d[key], cmap=BLUES, edgecolor=SURFACE, linewidth=0.6,
                        vmin=d[key].quantile(0.02), vmax=d[key].quantile(0.98))
        for code, r in o[o.n >= 50_000].nlargest(8, key).iterrows():  # 대형 공항 중 상위 8개
            ax.annotate(code, (r.lon, r.lat), xytext=(4, 3), textcoords="offset points", fontsize=8.5, color=INK)
        ax.set_aspect(1.25)
        ax.set_title(title + " — 연 5만 편 이상 공항 중 상위 8개 표시", fontsize=12)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, format=matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        cb.outline.set_visible(False)
    finish(fig, "07_airport_map", "공항별 결항·지연 비율 (미국 본토)",
           "원 크기 = 운항편 수 · 2022–2025 BTS 전체 · 연 5천 편 미만 공항, 알래스카·하와이·해외 영토 제외")


# ---------------------------------------------------------------- 8. 항공사
def fig_carrier():
    c = outcomes("hist_carrier").groupby("Reporting_Airline")[["n", "canc", "flown", "dly"]].sum()
    c["cancel"], c["delay"] = c.canc / c.n, c.dly / c.flown
    c = c.sort_values("delay")
    names = [f"{CARRIER.get(k, k)} ({k})" for k in c.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
    for ax, key, col, title in ((axes[0], "delay", C_DELAY, "지연(15분+) 비율"), (axes[1], "cancel", C_CANCEL, "결항 비율")):
        b = ax.barh(names, c[key], color=col, height=0.6)
        bar_labels(ax, b, c[key], horizontal=True)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, c[key].max() * 1.2)
        pct_axis(ax, "x")
    finish(fig, "08_carrier", "항공사별 지연·결항 비율",
           "2022–2025 BTS 전체 운항편 · 지연 비율 순 정렬 · 항공사는 그 자체로 강한 변수")


# ---------------------------------------------------------------- 9. Tail_Number 누수
def fig_tail_leak():
    df = load("merged", ["Tail_Number"])
    miss = df.Tail_Number.isna() | (df.Tail_Number.astype("string").str.strip() == "")
    df = df.assign(miss=miss.fillna(True).to_numpy(dtype=float))
    w = df.sample_weight
    g = (df.assign(mw=w * df.miss, w=w).groupby(["split", "label"])[["mw", "w"]].sum())
    g = (g.mw / g.w).unstack()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(SPLIT_NAME))
    width = 0.26
    for i, (lab, col) in enumerate(((0, C_NORMAL), (1, C_DELAY), (2, C_CANCEL))):
        v = g.reindex(list(SPLIT_NAME))[lab].values
        b = ax.bar(x + (i - 1) * width, v, width=width - 0.02, color=col, label=LABEL_NAME[lab])
        bar_labels(ax, b, v, fmt=lambda z: pct(z, 2))
    ax.set_xticks(x, list(SPLIT_NAME.values()))
    ax.grid(axis="x", visible=False)
    pct_axis(ax)
    ax.legend(loc="upper left")
    finish(fig, "09_tail_leak", "누수 발견: 결항편만 당일 기체 번호가 비어 있다",
           "v2의 당일 Tail_Number 기반 피처는 '결측 = 결항'을 알려주는 셈 → v3에서 과거 같은 편명 기준(tail_*_hist)으로 교체")


# ---------------------------------------------------------------- 10. 혼잡도 proxy vs 실제
CONG = {"origin_sched_deps_hour": "출발 공항 시간당 출발", "origin_sched_deps_day": "출발 공항 일 출발",
        "dest_sched_arrs_hour": "도착 공항 시간당 도착", "carrier_sched_deps_origin_day": "항공사 공항별 일 출발"}


def fig_proxy_vs_oracle():
    base = "origin_sched_deps_day"
    df = load("merged", [f"{base}_oracle", f"{base}_proxy_recent"])
    df = df[df.split == "test"].dropna()
    df = df[(df[f"{base}_oracle"] > 0) & (df[f"{base}_proxy_recent"] > 0)]  # 로그 축
    df = df.sample(n=min(400_000, len(df)), random_state=42)
    corr = json.loads((P / "interim" / "flight_stats.json").read_text(encoding="utf-8"))["congestion_corr"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    hb = ax.hexbin(df[f"{base}_oracle"], df[f"{base}_proxy_recent"], gridsize=60, cmap=BLUES, norm=LogNorm(),
                   mincnt=1, xscale="log", yscale="log", linewidths=0)
    lim = [df[[f"{base}_oracle", f"{base}_proxy_recent"]].min().min(),
           df[[f"{base}_oracle", f"{base}_proxy_recent"]].max().max()]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.set_xlabel("실제 당일 출발 편수 (서비스 시 알 수 없음)")
    ax.set_ylabel("proxy_recent (M-3월 같은 요일 평균)")
    ax.set_title(f"{CONG[base]} · 테스트 2025 · r = {corr[base + '|proxy_recent']:.3f}", fontsize=12)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.set_label("편 수")
    ax = axes[1]
    y = np.arange(len(CONG))
    for i, (kind, col, name) in enumerate((("proxy_ly", C_NORMAL, "작년 같은 요일"),
                                           ("proxy_recent", C_DELAY, "M-3월 같은 요일"))):
        v = [corr[f"{k}|{kind}"] for k in CONG]
        b = ax.barh(y + (0.5 - i) * 0.36, v, height=0.34, color=col, label=name)
        bar_labels(ax, b, v, fmt=lambda z: f"{z:.3f}", horizontal=True)
    ax.set_yticks(y, list(CONG.values()))
    ax.set_xlim(0.85, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("실제 편수와의 상관계수")
    ax.set_title("proxy 종류별 상관 (train)", fontsize=12)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.06), ncol=2)
    finish(fig, "10_proxy_vs_oracle", "서비스 시점에 구할 수 있는 값으로 혼잡도를 충분히 대체",
           "BTS는 약 3개월 늦게 공개되므로 당일 실제 편수 대신 공개된 과거 데이터로 만든 proxy를 사용")


# ---------------------------------------------------------------- 11. 데이터 규모
def fig_data_funnel():
    st = json.loads((P / "interim" / "flight_stats.json").read_text(encoding="utf-8"))
    ms = json.loads((P / "interim" / "merge_stats.json").read_text(encoding="utf-8"))
    steps = [("BTS 원본 (2020-11~2025-12, 과거 실적 집계용)", st["bts_rows_aggregated"]),
             ("모델 기간 운항편 (2022~2025)", st["flights_full"]),
             ("레이블 층화 샘플링 후", st["flights_sampled"]),
             ("기상 매칭 성공 → 학습셋 ml/", sum(ms["ml_rows"].values()))]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.grid(False)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    y = np.arange(len(steps))[::-1]
    v = np.array([n for _, n in steps], dtype=float)
    ax.barh(y, v, left=(v.max() - v) / 2, color=[C_NORMAL, C_DELAY, C_DELAY, "#184f95"], height=0.62)
    for yi, (name, n) in zip(y, steps):
        ax.text(v.max() / 2, yi, f"{n:,}", ha="center", va="center", color="white", fontsize=12,
                fontweight="bold")
        ax.text(v.max() * 1.01, yi, name, va="center", fontsize=10.5, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, v.max() * 1.55)
    split = " / ".join(f"{SPLIT_NAME[k].split()[0]} {n / 1e6:.2f}M" for k, n in ms["ml_rows"].items())
    finish(fig, "11_data_funnel", "데이터 규모: 3,400만 행에서 656만 행 학습셋으로", split)


FIGS = [fig_class_distribution, fig_monthly_trend, fig_flight_category, fig_weather_phenomena,
        fig_origin_dest_heatmap, fig_hourly, fig_airport_map, fig_carrier, fig_tail_leak, fig_proxy_vs_oracle,
        fig_data_funnel]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", help="그릴 그림 번호 (1부터)")
    args, _ = ap.parse_known_args()  # 주피터가 넘기는 --f=커널파일 인자는 무시
    for i, f in enumerate(FIGS, 1):
        if not args.only or i in args.only:
            f()
