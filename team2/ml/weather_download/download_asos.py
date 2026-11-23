"""IEM ASOS/AWOS METAR 아카이브를 공항 x 연도 단위로 순차 다운로드.

- station_map.csv 순서(BTS 운항 편수 많은 순)대로 진행
- 요청 1개씩, 요청 사이 1.5초 대기, 실패 시 지수 백오프로 최대 5회 재시도
- asos_raw/{bts_iata}/{bts_iata}_{YYYY}.csv.gz 이 이미 있으면 건너뜀 (재실행 시 이어받기)
- 응답이 비었거나 헤더만 있으면 파일을 만들지 않고 실패로 기록
- 결과는 download_log.csv 에 누적 기록

사용: python ml/weather_download/download_asos.py [--only ATL,DFW] [--years 2020,2021]
"""
import argparse
import csv
import gzip
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

def _weather_root():
    """날씨 데이터 폴더(station_map.csv, asos_raw/ 가 있는 곳, 저장소에는 없음).

    1) 환경변수 AIRLINE_DATA_ROOT/weather  2) 이 파일의 상위 폴더 중 weather/station_map.csv 가 있는 곳
    3) 없으면 현재 폴더/weather (처음 받을 때)
    """
    if os.environ.get("AIRLINE_DATA_ROOT"):
        return Path(os.environ["AIRLINE_DATA_ROOT"]).resolve() / "weather"
    for d in Path(__file__).resolve().parents:
        if (d / "weather" / "station_map.csv").is_file():
            return d / "weather"
    return Path.cwd() / "weather"


ROOT = _weather_root()
RAW = ROOT / "asos_raw"
LOG = ROOT / "download_log.csv"
UA = "bts-weather-research/1.0 (python-urllib)"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

DATA_COLS = [
    "tmpf", "dwpf", "relh", "feel", "drct", "sknt", "gust", "peak_wind_gust", "vsby",
    "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4",
    "p01i", "wxcodes", "ice_accretion_1hr", "ice_accretion_3hr", "ice_accretion_6hr",
    "snowdepth", "alti", "mslp", "metar",
]
YEARS = list(range(2020, 2026))
# 첫 해는 2019-12-31, 마지막 해는 2026-01-01까지 포함 (ets는 배타적 끝)
PERIOD_START, PERIOD_END = "2019-12-31T00:00Z", "2026-01-02T00:00Z"
SLEEP_S = 1.5
MAX_TRIES = 5
LOG_COLS = ["bts_iata", "iem_station", "year", "rows", "status", "error", "downloaded_at"]


def year_window(year):
    sts = PERIOD_START if year == YEARS[0] else f"{year}-01-01T00:00Z"
    ets = PERIOD_END if year == YEARS[-1] else f"{year + 1}-01-01T00:00Z"
    return sts, ets


def build_url(station, year):
    sts, ets = year_window(year)
    params = [
        ("station", station), ("data", ",".join(DATA_COLS)),
        ("sts", sts), ("ets", ets), ("tz", "Etc/UTC"), ("format", "onlycomma"),
        ("latlon", "yes"), ("elev", "yes"), ("missing", "empty"), ("trace", "T"),
        ("report_type", "3,4"),
    ]
    return f"{URL}?{urllib.parse.urlencode(params)}"


def fetch(url):
    """(body, error). 429/5xx/네트워크 오류는 지수 백오프 재시도."""
    err = ""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read(), ""
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace').strip()}"
            if e.code not in (429, 500, 502, 503, 504):
                return None, err
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        if attempt < MAX_TRIES:
            wait = 5 * 2 ** (attempt - 1)  # 5, 10, 20, 40s
            print(f"    try {attempt} failed ({err}); sleep {wait}s", flush=True)
            time.sleep(wait)
    return None, f"{MAX_TRIES} tries: {err}"


def log_row(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="쉼표로 구분한 bts_iata 목록")
    ap.add_argument("--years", help="쉼표로 구분한 연도 목록")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    years = [int(y) for y in args.years.split(",")] if args.years else YEARS

    with open(ROOT / "station_map.csv", encoding="utf-8") as fh:
        stations = [r for r in csv.DictReader(fh) if r["iem_station"]]
    if only:
        stations = [r for r in stations if r["bts_iata"] in only]

    n_done = n_new = n_fail = 0
    for i, st in enumerate(stations, 1):
        iata, sid = st["bts_iata"], st["iem_station"]
        for year in years:
            out = RAW / iata / f"{iata}_{year}.csv.gz"
            if out.exists():
                n_done += 1
                continue
            t0 = time.time()
            body, err = fetch(build_url(sid, year))
            rows, status = 0, "failed"
            if body is not None:
                lines = [ln for ln in body.decode("utf-8", "replace").splitlines() if ln.strip()]
                rows = max(len(lines) - 1, 0)
                if not lines or not lines[0].startswith("station,valid"):
                    err = "unexpected response: " + (lines[0][:200] if lines else "empty body")
                elif rows == 0:
                    err = "header only (0 rows)"
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".gz.part")
                    with gzip.open(tmp, "wb", compresslevel=6) as gz:
                        gz.write(body)
                    os.replace(tmp, out)
                    status = "ok"
            if status == "ok":
                n_new += 1
            else:
                n_fail += 1
            log_row(dict(bts_iata=iata, iem_station=sid, year=year, rows=rows, status=status,
                         error=err, downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
            print(f"[{i}/{len(stations)}] {iata}({sid}) {year}: {status} rows={rows} "
                  f"{time.time() - t0:.1f}s {err}", flush=True)
            time.sleep(SLEEP_S)
    print(f"done. skipped(existing)={n_done} new={n_new} failed={n_fail}")


if __name__ == "__main__":
    main()
