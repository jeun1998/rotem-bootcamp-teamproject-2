"""IEM TAF(공항 예보) 아카이브를 공항 x 연도 단위로 순차 다운로드 (선택 작업).

download_asos.py 와 같은 규칙(순차, 1.5초 간격, 백오프 5회, 이어받기)을 따른다.
sts/ets 는 TAF '발표 시각' 범위. 출력: taf_raw/{bts_iata}/{bts_iata}_{YYYY}.csv.gz, taf_download_log.csv
TAF를 발표하지 않는 공항은 헤더만 오므로 status=no_taf 로 기록한다.

사용: python ml/weather_download/download_taf.py [--only ATL,DFW] [--years 2020]
"""
import argparse
import csv
import gzip
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I 로 실행해도 import 되도록
from download_asos import LOG_COLS, SLEEP_S, YEARS, fetch, year_window

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
RAW = ROOT / "taf_raw"
LOG = ROOT / "taf_download_log.csv"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py"


def log_row(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--years")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    years = [int(y) for y in args.years.split(",")] if args.years else YEARS

    with open(ROOT / "station_map.csv", encoding="utf-8") as fh:
        stations = [r for r in csv.DictReader(fh) if r["iem_station"]]
    if only:
        stations = [r for r in stations if r["bts_iata"] in only]

    # 이전 실행에서 TAF가 없던 공항·연도는 다시 묻지 않는다
    no_taf = set()
    if LOG.exists():
        with open(LOG, encoding="utf-8") as fh:
            no_taf = {(r["bts_iata"], r["year"]) for r in csv.DictReader(fh) if r["status"] == "no_taf"}

    for i, st in enumerate(stations, 1):
        iata, sid = st["bts_iata"], st["iem_station"]
        for year in years:
            out = RAW / iata / f"{iata}_{year}.csv.gz"
            if out.exists() or (iata, str(year)) in no_taf:
                continue
            sts, ets = year_window(year)
            url = f"{URL}?" + urllib.parse.urlencode(
                [("station", sid), ("sts", sts), ("ets", ets), ("tz", "Etc/UTC"), ("fmt", "csv")])
            t0 = time.time()
            body, err = fetch(url)
            rows, status = 0, "failed"
            if body is not None:
                lines = [ln for ln in body.decode("utf-8", "replace").splitlines() if ln.strip()]
                rows = max(len(lines) - 1, 0)
                if not lines or "," not in lines[0]:
                    err = "unexpected response: " + (lines[0][:200] if lines else "empty body")
                elif rows == 0:
                    status, err = "no_taf", "header only (0 rows)"
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".gz.part")
                    with gzip.open(tmp, "wb", compresslevel=6) as gz:
                        gz.write(body)
                    os.replace(tmp, out)
                    status = "ok"
            log_row(dict(bts_iata=iata, iem_station=sid, year=year, rows=rows, status=status, error=err,
                         downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
            print(f"[{i}/{len(stations)}] {iata}({sid}) {year}: {status} rows={rows} "
                  f"{time.time() - t0:.1f}s {err}", flush=True)
            time.sleep(SLEEP_S)


if __name__ == "__main__":
    main()
