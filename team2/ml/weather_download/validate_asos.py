"""asos_raw/ 검증: coverage_report.csv 생성 + 요약 출력 (README 작성용).

coverage_pct = 해당 UTC 연도(1/1 00시 ~ 12/31 23시)의 시간 슬롯 중 관측이 1건 이상 있는 비율.
여유분(2019-12-31, 2026-01-01)은 분모·분자에서 제외하고 행 수(rows)에만 포함.
"""
import csv
import gzip
import json
from collections import Counter
from datetime import datetime
import os
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
YEARS = list(range(2020, 2026))
META_COLS = {"station", "valid", "lon", "lat", "elevation"}


def main():
    with open(ROOT / "station_map.csv", encoding="utf-8") as fh:
        smap = list(csv.DictReader(fh))

    report, empty, filled = [], Counter(), Counter()
    total_rows = total_bytes = 0
    for st in smap:
        iata = st["bts_iata"]
        for year in YEARS:
            path = RAW / iata / f"{iata}_{year}.csv.gz"
            rec = dict(bts_iata=iata, iem_station=st["iem_station"], year=year, rows=0, rows_in_year=0,
                       hours_in_year=0, hours_with_obs=0, coverage_pct=0.0, first_valid="", last_valid="",
                       file_exists=path.exists())
            hours_total = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days * 24
            rec["hours_in_year"] = hours_total
            if path.exists():
                total_bytes += path.stat().st_size
                hours = set()
                with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
                    rdr = csv.DictReader(fh)
                    for r in rdr:
                        rec["rows"] += 1
                        for k, v in r.items():
                            if k in META_COLS:
                                continue
                            (filled if v not in ("", None) else empty)[k] += 1
                        v = r["valid"]
                        if v.startswith(str(year)):
                            rec["rows_in_year"] += 1
                            hours.add(v[:13])
                        rec["first_valid"] = rec["first_valid"] or v
                        rec["last_valid"] = v
                rec["hours_with_obs"] = len(hours)
                rec["coverage_pct"] = round(100 * len(hours) / hours_total, 2)
                total_rows += rec["rows"]
            report.append(rec)

    cols = list(report[0].keys())
    with open(ROOT / "coverage_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(report)

    low = [r for r in report if r["coverage_pct"] < 80]
    missing_pct = {k: round(100 * empty[k] / (empty[k] + filled[k]), 2) for k in empty | filled}
    summary = {
        "files": sum(r["file_exists"] for r in report),
        "expected_files": len(report),
        "total_rows": total_rows,
        "raw_gz_bytes": total_bytes,
        "coverage_lt80": [(r["bts_iata"], r["year"], r["coverage_pct"]) for r in low],
        "missing_pct_by_col": dict(sorted(missing_pct.items(), key=lambda kv: -kv[1])),
        "coverage_mean": round(sum(r["coverage_pct"] for r in report) / len(report), 2),
    }
    (ROOT / "meta" / "validation_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "coverage_lt80"}, indent=1))
    print(f"coverage < 80%: {len(low)} airport-years")
    for r in low:
        print(f"  {r['bts_iata']} {r['year']} {r['coverage_pct']}% rows={r['rows']}")


if __name__ == "__main__":
    main()
