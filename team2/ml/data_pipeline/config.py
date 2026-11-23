"""공통 경로·상수. 원본(flight_expectation/, weather/)은 읽기만 하고 결과는 processed_v2/ 에만 쓴다.

v2: 연 단위 분할 + train 정상편 다운샘플링 + 도착 공항 기상 + Parquet 출력.
v3: BTS 공개 지연(BTS_LAG_MONTHS) 규칙 적용 — hist·혼잡도·기체 일정 피처를 서비스 시점에 공개된 BTS로만 계산.
    기상 피처는 v2 결과를 flight_id로 재사용한다. processed/(v1), processed_v2/ 는 비교용으로 그대로 둔다.
"""
import json
import os
import sys
import time
from pathlib import Path



REPO = Path(__file__).resolve().parents[2]       # flight-delay-predictor/


def _data_root():
    """데이터 폴더 = flight_expectation/, weather/, processed_v3/ 가 있는 곳 (저장소에는 없음).

    1) 환경변수 AIRLINE_DATA_ROOT  2) 이 파일의 상위 폴더 중 weather/station_map.csv 가 있는 곳
    3) 없으면 저장소 폴더 — 파이프라인은 못 돌리지만 service_features 는 저장소의 집계표로 동작
    """
    env = os.environ.get("AIRLINE_DATA_ROOT")
    if env:
        return Path(env).resolve()
    for d in Path(__file__).resolve().parents:
        if (d / "weather" / "station_map.csv").is_file():
            return d
    return REPO


ROOT = _data_root()
BTS_DIR = ROOT / "flight_expectation" / "flight_expectation" / "data" / "bts_selected"
WEATHER_DIR = ROOT / "weather"
ASOS_DIR = WEATHER_DIR / "asos_raw"
STATION_MAP = WEATHER_DIR / "station_map.csv"
if not STATION_MAP.is_file():                    # 데이터 폴더가 없으면 저장소 사본 사용
    STATION_MAP = REPO / "ml" / "weather_download" / "station_map.csv"

INSPECTION = ROOT / "processed" / "inspection"   # 원본 탐색 결과 (v1에서 생성, 그대로 재사용)

V2 = ROOT / "processed_v2"                       # 기상 결합 결과 재사용 (읽기 전용)
V2_MERGED = V2 / "merged"
PROCESSED = ROOT / "processed_v3"
INTERIM = PROCESSED / "interim"                  # 중간 산출물 (Parquet, 재생성 가능)
AGG = PROCESSED / "aggregates"                   # 전체 BTS로 만든 집계표 (웹 서비스 재사용)
if not AGG.is_dir():                             # 데이터 폴더가 없으면 저장소에 커밋된 집계표 사용
    AGG = REPO / "data" / "aggregates"
WX_INTERIM = V2 / "interim" / "weather"         # v2 기상 특징 (year=YYYY/month=MM/batch-XX.parquet)
FLIGHT_INTERIM = INTERIM / "flights"             # split=*/year=YYYY/month=MM/part.parquet (샘플 후)
MERGED = PROCESSED / "merged"
ML = PROCESSED / "ml"
RUN_LOG = PROCESSED / "run_log.jsonl"

# ---------------------------------------------------------------- 분할·샘플링
SPLITS = {"train": ("2022-01-01", "2023-12-31"),
          "validation": ("2024-01-01", "2024-12-31"),
          "test": ("2025-01-01", "2025-12-31")}
NEG_RATE = 0.2       # train 정상(0)편 추출 비율. 지연·결항은 전부 유지
EVAL_RATE = 0.1      # validation/test 균등 추출 비율 (클래스 분포 유지)
SAMPLE_SEED = 42
# 2022-01 운항편의 hist 창(2020-11~2021-10)과 작년 같은 요일 참조(D-371)를 위해 2020-11부터 읽는다
BTS_READ_START = "2020-11-01"
# BTS 공개 지연: 운항월 M이면 M-BTS_LAG_MONTHS 월 말까지의 BTS만 사용 가능 (보수적으로 3개월)
BTS_LAG_MONTHS = 3

# ---------------------------------------------------------------- 시점·결합
PREDICTION_LEAD_H = 3        # 예측 시점 = 예정 출발 3시간 전
WEATHER_TOLERANCE_H = 2      # 예측 시점 이전 최대 2시간 이내 관측만 사용
OBS_PUBLISH_LAG_MIN = 0      # METAR 공개 지연 가정 (분). 공개 시각 미확인 → 0, 보고서에 한계 명시
DELAY_THRESHOLD_MIN = 15
WX_MONTH_MARGIN_DAYS = 2     # 월 처리 시 앞뒤로 불러오는 기상 여유 (시차 ±14h + 리드 3h + 허용 2h)
WX_STATION_BATCH = 25        # 기상 전처리 시 한 번에 처리하는 관측소 수 (메모리 상한)

# 과거 실적: 운항월 M 기준 M-14 ~ M-3 (12개월, 공개 지연 반영), 베이지안 평활
HIST_WINDOW_MONTHS = 12
HIST_SMOOTH_M = 50
# 혼잡도 proxy
CONGESTION_LAGS_DAYS = (7, 14, 21, 28)   # v2 방식 (서비스 시 사용 불가 → *_proxy_v2_unavailable)
PROXY_LY_LAGS_DAYS = (364, 371, 357)     # 작년 같은 요일과 앞뒤 1주
TAIL_LY_LAGS_DAYS = (364, 371, 357)      # tail_*_hist 참조일 (+ M-3월 같은 요일)

# IEM 메타데이터 시간대 오류 보정 (버진아일랜드는 서머타임 없는 AST)
TZ_OVERRIDES = {"STT": "America/St_Thomas", "STX": "America/St_Thomas"}

PARQUET_KW = dict(engine="pyarrow", compression="snappy", index=False)


def source_month(flight_month):
    """운항월(Period 'M' 또는 'YYYY-MM')에 대해 사용할 수 있는 마지막 BTS 월 (M - BTS_LAG_MONTHS)."""
    import pandas as pd
    return pd.Period(flight_month, freq="M") - BTS_LAG_MONTHS


def ensure_dirs():
    for d in (PROCESSED, INTERIM, AGG, FLIGHT_INTERIM, MERGED, ML):
        d.mkdir(parents=True, exist_ok=True)


def split_of_month(year, month):
    ym = f"{year}-{month:02d}"
    for name, (a, b) in SPLITS.items():
        if a[:7] <= ym <= b[:7]:
            return name
    return None


def dataset_months():
    import pandas as pd
    start = min(a for a, _ in SPLITS.values())
    end = max(b for _, b in SPLITS.values())
    return [(p.year, p.month) for p in pd.period_range(start, end, freq="M")]


def peak_memory_mb():
    """현재 프로세스의 최대 메모리 사용량(MB). Windows: PeakWorkingSetSize."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


class RunTimer:
    """스크립트별 처리 시간·최대 메모리를 run_log.jsonl 에 남긴다."""

    def __init__(self, name):
        self.name, self.t0 = name, time.time()

    def done(self, **extra):
        rec = dict(script=self.name, seconds=round(time.time() - self.t0, 1), peak_memory_mb=peak_memory_mb(),
                   finished_at=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
        with open(RUN_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{self.name}] {rec['seconds']}s, peak {rec['peak_memory_mb']} MB")
