# data/aggregates/

`ml/data_pipeline/preprocess_flights.py`가 BTS 전체 운항편(2020-11 ~ 2025-12)으로 만든 집계표.
학습셋의 과거 실적·혼잡도·기체 일정 피처를 만드는 데 쓰였고, 서비스에서도 같은 피처를
`ml/data_pipeline/service_features.py`의 `build_service_features()`로 다시 만들 때 읽습니다.
데이터 폴더(`AIRLINE_DATA_ROOT/processed_v3/aggregates`)가 없으면 이 폴더를 대신 사용합니다.

| 파일 | 내용 | 키 |
|---|---|---|
| `monthly_outcomes_hist_*.parquet` | 월별 운항·결항·지연 편수 (`n`, `canc`, `flown`, `dly`) | `ym` + 그룹 |
| `hist_*.parquet` | 과거 12개월(M−14~M−3) 지연률·결항률, 베이지안 평활(m=50) | 운항월 + 그룹 |
| `sched_counts_*.parquet` | 일·시간별 예정 편수 (혼잡도 proxy 계산용) | 날짜(+시) + 공항/항공사 |
| `route_distance.parquet` | 노선별 거리 | `Origin`, `Dest` |
| `tail_sameday/year=YYYY/month=MM/part.parquet` | 편명별 당일 기체 일정 (기체 일정 피처 참조용) | 항공사·편명·출발 공항·날짜 |

그룹(`*`): `carrier`, `origin`, `dest`, `route`, `origin_hour`, `carrier_origin`.

- BTS는 약 3개월 늦게 공개되므로 운항월 M의 피처는 M−3월까지의 데이터만 씁니다 (`config.BTS_LAG_MONTHS`).
- 총 약 196MB. 2026년 이후 서비스에 쓰려면 새 BTS 데이터로 `preprocess_flights.py`를 다시 돌려 갱신해야 합니다.
