"""v3 병합: v2 병합본(기상 결합 완료)에 v3 BTS 쪽 피처를 flight_id로 붙인다. 기상은 다시 계산하지 않는다.

v2 → v3 컬럼 처리
  - hist_*            : v2 값 삭제, v3 값(M-14~M-3 창)으로 교체
  - *_proxy           : *_proxy_v2_unavailable 로 이름 변경 (병합본 전용, 학습 제외)
  - tail_* (당일 기준) : *_sameday 로 이름 변경 (병합본 전용, 학습 제외)
  - 추가              : *_proxy_ly, *_proxy_recent, origin_yoy_ratio, tail_*_hist, bts_source_max/limit_date
입력: processed_v2/merged/, processed_v3/interim/flights/
출력: processed_v3/merged/, ml/, unmatched_flights.parquet, feature_lists.json, interim/merge_stats.json
사용: python ml/data_pipeline/merge_datasets.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import column_selection as S  # noqa: E402
import config as C  # noqa: E402

SAMEDAY_RENAME = {"tail_leg_index_of_day": "tail_leg_index_of_day_sameday",
                  "tail_sched_turn_min": "tail_sched_turn_min_sameday",
                  "tail_prev_origin": "tail_prev_origin_sameday",
                  "tail_prev_sched_elapsed": "tail_prev_sched_elapsed_sameday"}
TOL = pd.Timedelta(hours=C.WEATHER_TOLERANCE_H)


def choose_proxy():
    """oracle 과의 평균 상관이 더 높은 proxy 를 base_plus_candidates 에 쓴다."""
    corr = json.loads((C.INTERIM / "flight_stats.json").read_text(encoding="utf-8"))["congestion_corr"]
    mean = {p: np.mean([v for k, v in corr.items() if k.endswith(f"|{p}")]) for p in ("proxy_ly", "proxy_recent")}
    return max(mean, key=mean.get), mean


def main():
    timer = C.RunTimer("merge_datasets")
    C.ensure_dirs()
    proxy, proxy_mean = choose_proxy()
    lists = S.feature_lists(proxy)
    ml_features = list(dict.fromkeys(f for k, v in lists.items() for f in v))
    forbidden = set(S.forbidden_columns())
    assert not forbidden & set(ml_features), forbidden & set(ml_features)

    stats = dict(rows=Counter(), ml_rows=Counter(), id_equal_months=0, id_diff_months=[], label_mismatch=0,
                 leak_origin_future=0, leak_dest_future=0, leak_origin_over_tol=0, leak_dest_over_tol=0,
                 timing_violation_rows=0, proxy_choice=proxy, proxy_mean_corr=proxy_mean)
    unmatched = []
    for year, month in C.dataset_months():
        split = C.split_of_month(year, month)
        sub = f"split={split}/year={year}/month={month:02d}/part.parquet"
        v2 = pd.read_parquet(C.V2_MERGED / sub)
        new = pd.read_parquet(C.FLIGHT_INTERIM / sub)
        if set(v2["flight_id"]) == set(new["flight_id"]):
            stats["id_equal_months"] += 1
        else:
            stats["id_diff_months"].append(f"{year}-{month:02d}")
        chk = v2[["flight_id", "label"]].merge(new[["flight_id", "label"]], on="flight_id", suffixes=("", "_v3"))
        stats["label_mismatch"] += int((chk["label"] != chk["label_v3"]).sum())
        new = new.drop(columns=["label"])
        v2 = v2.drop(columns=[c for c in v2.columns if c.startswith("hist_")])
        v2 = v2.rename(columns={**SAMEDAY_RENAME,
                                **{f"{n}_proxy": f"{n}_proxy_v2_unavailable" for n in S.CONGESTION_NAMES}})
        df = v2.merge(new, on="flight_id", how="left", validate="one_to_one")
        del v2, new

        m = df["weather_match_status"] == "matched"
        md = df["dest_weather_match_status"] == "matched"
        stats["leak_origin_future"] += int((df.loc[m, "wx_obs_valid"] > df.loc[m, "prediction_time_utc"]).sum())
        stats["leak_dest_future"] += int((df.loc[md, "dest_obs_valid"] > df.loc[md, "prediction_time_utc"]).sum())
        stats["leak_origin_over_tol"] += int((df.loc[m, "wx_obs_valid"] < df.loc[m, "prediction_time_utc"] - TOL).sum())
        stats["leak_dest_over_tol"] += int((df.loc[md, "dest_obs_valid"] < df.loc[md, "prediction_time_utc"] - TOL).sum())
        stats["timing_violation_rows"] += int((df["bts_source_max_date"] > df["bts_source_limit_date"]).sum())
        stats["rows"][split] += len(df)

        out = C.MERGED / f"split={split}" / f"year={year}" / f"month={month:02d}"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "part.parquet", **C.PARQUET_KW)
        ml = df.loc[df["in_ml_dataset"], ["flight_id"] + ml_features + [S.LABEL] + S.SAMPLING_COLS]
        (C.ML / f"split={split}").mkdir(parents=True, exist_ok=True)
        ml.to_parquet(C.ML / f"split={split}" / f"part-{year}-{month:02d}.parquet", **C.PARQUET_KW)
        stats["ml_rows"][split] += len(ml)
        unmatched.append(df.loc[~df["in_ml_dataset"]].assign(split=split))
        print(f"  {year}-{month:02d} {split}: {len(df):,} rows, ml {len(ml):,} (peak {C.peak_memory_mb()} MB)",
              flush=True)
        del df, ml

    pd.concat(unmatched, ignore_index=True).to_parquet(C.PROCESSED / "unmatched_flights.parquet", **C.PARQUET_KW)
    lists_out = dict(label=S.LABEL, id="flight_id", categorical=S.CATEGORICAL, sampling_columns=S.SAMPLING_COLS,
                     **lists, candidates_only=S.candidate_features(proxy), dest_weather_only=S.DEST_WEATHER_FEATURES,
                     tail_only=S.TAIL_FEATURES, forbidden_in_features=sorted(forbidden),
                     congestion_proxy_in_base_plus=proxy,
                     notes=("experimental_tail = with_dest_weather + tail_*_hist 5개. "
                            "alt_congestion_candidates 는 base_plus_candidates 의 혼잡도 proxy 와 바꿔 실험할 대안. "
                            "sample_rate/sample_weight는 학습 특징이 아님(가중치로만 사용). "
                            "BTS 기반 피처는 운항월 M 에 대해 M-3월 말까지 공개된 BTS만 사용."))
    (C.PROCESSED / "feature_lists.json").write_text(json.dumps(lists_out, indent=1, ensure_ascii=False),
                                                    encoding="utf-8")
    stats = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
    (C.INTERIM / "merge_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False, default=float),
                                                encoding="utf-8")
    timer.done(rows=sum(stats["rows"].values()), ml_rows=sum(stats["ml_rows"].values()), proxy=proxy)


if __name__ == "__main__":
    main()
