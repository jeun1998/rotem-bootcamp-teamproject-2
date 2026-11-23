# 데이터 품질 검증 보고서 (v3)

v2(`processed_v2/`)에서 두 문제를 고친 v3입니다. (1) 당일 Tail_Number로 만든 `tail_*`의 결측이 곧 결항이라는 누수, (2) BTS가 약 2개월 늦게 공개돼 서비스 시점에 만들 수 없는 `*_proxy`(D−7~28)와 `hist_*`(직전 달까지). 이제 운항월 M의 BTS 기반 피처는 **M−3월 말까지 공개된 BTS만** 씁니다. 샘플링(같은 seed)과 기상 결합은 v2 결과를 그대로 재사용했고, BTS 쪽 피처만 월 단위로 다시 계산했습니다. 레이블, 예측 시점, merge_asof, UTC 변환, 기상 QC, 3시간 창 특징은 v1·v2와 같습니다.

## 1. 환경과 설정

| item | value |
|---|---|
| RAM 전체 (GB) | 15.2 |
| RAM 작업 시작 시 사용 가능 (GB) | 4.9 |
| 디스크 여유 (GB, 현재) | 380.9 |
| train 기간 | 2022-01-01 ~ 2023-12-31 |
| validation 기간 | 2024-01-01 ~ 2024-12-31 |
| test 기간 | 2025-01-01 ~ 2025-12-31 |
| NEG_RATE | 0.2 |
| EVAL_RATE | 0.1 |
| SAMPLE_SEED | 42 (월별 seed = SAMPLE_SEED + YYYYMM) |
| BTS_LAG_MONTHS | 3 (운항월 M → M−3월 말까지 BTS 사용) |
| hist 창 | M−14 ~ M−3 (12개월, 월 단위), 평활 m=50 |
| 혼잡도 proxy_ly | D−364/371/357 평균 |
| 혼잡도 proxy_recent | M−3월 중 같은 요일인 날들의 평균 |
| base_plus_candidates 혼잡도 | proxy_recent |
| 기상 월 여유 | ±2일 |
| 최대 메모리 (모든 단계 중) | 3,429 MB |

- 사전 추정: 작업 시작 시 사용 가능 메모리가 4.9GB라, 전체를 한 번에 올리는 방식(수십 GB)은 불가능했습니다. 그래서 BTS는 월 단위로 필요한 컬럼만, 기상은 관측소 25개씩, 결합은 월 단위로 처리하도록 설계했습니다. 디스크는 여유가 충분합니다. 이 방식으로 기본값(NEG_RATE=0.2, train 2022~2023)이 감당되어 **설정값을 바꾸지 않았습니다.**
- v3는 기상 전처리·결합을 다시 하지 않았습니다(아래 preprocess_weather는 v2 실행 기록).

| script | seconds | minutes | peak_memory_mb |
|---|---|---|---|
| preprocess_weather (v2, 재사용) | 169.1 | 2.8 | 2,159.6 |
| preprocess_flights | 444.9 | 7.4 | 1,783.6 |
| merge_datasets | 51.4 | 0.9 | 1,222.3 |
| validate_dataset | 94.7 | 1.6 | 3429 |

| path | size_mb |
|---|---|
| merged/ | 1,099.2 |
| ml/ | 377.1 |
| aggregates/ | 195.7 |
| interim/ (재생성 가능) | 217.1 |
| unmatched_flights.parquet | 1.1 |

## 2. 규모 (샘플링 전 → 후 → 학습셋)

| split | n_full | n_sample | ml_rows | period |
|---|---|---|---|---|
| test | 7,001,619 | 700,161 | 699,382 | 2025-01-01 ~ 2025-12-31 |
| train | 13,577,024 | 5,156,768 | 5,153,143 | 2022-01-01 ~ 2023-12-31 |
| validation | 7,079,061 | 707,906 | 707,279 | 2024-01-01 ~ 2024-12-31 |

- v2와 flight_id 집합이 같은 달: 48 / 48. v2와 label이 다른 행: 0.
- `ml_rows`는 출발 공항 기상 매칭에 성공하고 Label이 정의된 행입니다.

## 3. 샘플링 전후 클래스 분포

split 합계 기준입니다. '가중 복원'은 샘플 행에 `sample_weight`를 곱해 되돌린 분포입니다.

| split | 원본 0 정상 % | 원본 1 지연(≥15분) % | 원본 2 결항 % | 샘플 0 정상 % | 샘플 1 지연(≥15분) % | 샘플 2 결항 % | 가중 복원 0 정상 % | 가중 복원 1 지연(≥15분) % | 가중 복원 2 결항 % |
|---|---|---|---|---|---|---|---|---|---|
| test | 77.119 | 21.412 | 1.469 | 77.119 | 21.412 | 1.469 | 77.119 | 21.412 | 1.469 |
| train | 77.523 | 20.494 | 1.983 | 40.821 | 53.958 | 5.221 | 77.523 | 20.494 | 1.983 |
| validation | 78.282 | 20.358 | 1.361 | 78.282 | 20.358 | 1.36 | 78.282 | 20.358 | 1.36 |

- 가중 복원 분포와 원본 분포의 최대 차이: **0.001%p**.
- validation/test 샘플과 원본의 클래스 비율 최대 차이: **0.001%p** (기준 ±0.1%p 이내: 통과).

### split × 월별

| split | month | rows_full | rows_sample | full_0% | sample_0% | full_1% | sample_1% | full_2% | sample_2% |
|---|---|---|---|---|---|---|---|---|---|
| test | 2025-01 | 539,747 | 53,974 | 79.3 | 79.3 | 17.68 | 17.68 | 3.02 | 3.02 |
| test | 2025-02 | 504,884 | 50,488 | 79.19 | 79.19 | 19.35 | 19.35 | 1.47 | 1.47 |
| test | 2025-03 | 600,872 | 60,087 | 79.91 | 79.91 | 18.93 | 18.93 | 1.15 | 1.15 |
| test | 2025-04 | 583,950 | 58,395 | 80.61 | 80.61 | 18.55 | 18.55 | 0.84 | 0.84 |
| test | 2025-05 | 605,648 | 60,564 | 76.69 | 76.69 | 22.26 | 22.26 | 1.05 | 1.05 |
| test | 2025-06 | 611,575 | 61,158 | 71.41 | 71.41 | 27 | 27 | 1.59 | 1.59 |
| test | 2025-07 | 631,428 | 63,143 | 69.69 | 69.69 | 27.86 | 27.86 | 2.45 | 2.45 |
| test | 2025-08 | 602,378 | 60,238 | 76.83 | 76.83 | 22.1 | 22.1 | 1.08 | 1.08 |
| test | 2025-09 | 562,439 | 56,244 | 83.08 | 83.08 | 16.41 | 16.41 | 0.51 | 0.51 |
| test | 2025-10 | 605,844 | 60,584 | 79.98 | 79.98 | 19.49 | 19.49 | 0.53 | 0.53 |
| test | 2025-11 | 570,550 | 57,055 | 78.11 | 78.11 | 19.4 | 19.4 | 2.49 | 2.49 |
| test | 2025-12 | 582,304 | 58,231 | 72.01 | 72.01 | 26.45 | 26.45 | 1.55 | 1.55 |
| train | 2022-01 | 537,902 | 213,148 | 75.47 | 38.09 | 18.35 | 46.31 | 6.18 | 15.6 |
| train | 2022-02 | 495,713 | 191,470 | 76.72 | 39.72 | 18.9 | 48.92 | 4.38 | 11.35 |
| train | 2022-03 | 564,853 | 215,928 | 77.22 | 40.4 | 21.24 | 55.55 | 1.55 | 4.05 |
| train | 2022-04 | 556,502 | 219,152 | 75.77 | 38.48 | 21.88 | 55.57 | 2.34 | 5.95 |
| train | 2022-05 | 578,819 | 221,562 | 77.15 | 40.31 | 20.88 | 54.55 | 1.97 | 5.14 |
| train | 2022-06 | 577,283 | 241,085 | 72.8 | 34.86 | 24.11 | 57.72 | 3.1 | 7.41 |
| train | 2022-07 | 594,957 | 240,488 | 74.47 | 36.85 | 23.75 | 58.77 | 1.77 | 4.39 |
| train | 2022-08 | 589,810 | 235,264 | 75.14 | 37.68 | 22.42 | 56.2 | 2.44 | 6.12 |
| train | 2022-09 | 557,494 | 193,263 | 81.67 | 47.12 | 16.77 | 48.38 | 1.56 | 4.5 |
| train | 2022-10 | 572,287 | 193,311 | 82.78 | 49.01 | 16.38 | 48.49 | 0.84 | 2.5 |
| train | 2022-11 | 546,410 | 193,997 | 80.62 | 45.41 | 18.25 | 51.4 | 1.13 | 3.19 |
| train | 2022-12 | 557,095 | 249,057 | 69.12 | 30.92 | 25.39 | 56.8 | 5.49 | 12.28 |
| train | 2023-01 | 538,837 | 205,227 | 77.39 | 40.64 | 20.7 | 54.34 | 1.91 | 5.02 |
| train | 2023-02 | 502,749 | 180,779 | 80.05 | 44.53 | 18.15 | 50.49 | 1.79 | 4.99 |
| train | 2023-03 | 580,322 | 225,331 | 76.46 | 39.39 | 22.26 | 57.33 | 1.28 | 3.29 |
| train | 2023-04 | 561,441 | 218,835 | 76.28 | 39.14 | 22.01 | 56.48 | 1.71 | 4.38 |
| train | 2023-05 | 579,958 | 204,105 | 81.01 | 46.04 | 18.42 | 52.34 | 0.57 | 1.62 |
| train | 2023-06 | 577,262 | 246,737 | 71.57 | 33.49 | 26.31 | 61.56 | 2.12 | 4.95 |
| train | 2023-07 | 601,866 | 267,428 | 69.46 | 31.26 | 28.11 | 63.27 | 2.43 | 5.46 |
| train | 2023-08 | 602,987 | 233,180 | 76.66 | 39.65 | 21.82 | 56.42 | 1.52 | 3.93 |
| train | 2023-09 | 569,338 | 202,740 | 80.49 | 45.21 | 18.27 | 51.31 | 1.24 | 3.48 |
| train | 2023-10 | 598,968 | 197,478 | 83.79 | 50.83 | 15.85 | 48.09 | 0.36 | 1.08 |
| train | 2023-11 | 563,777 | 175,971 | 85.98 | 55.1 | 13.87 | 44.44 | 0.15 | 0.47 |
| train | 2023-12 | 570,394 | 191,232 | 83.09 | 49.57 | 16.5 | 49.22 | 0.4 | 1.21 |
| validation | 2024-01 | 547,271 | 54,727 | 73.99 | 73.99 | 22.28 | 22.28 | 3.73 | 3.73 |
| validation | 2024-02 | 519,221 | 51,922 | 84.02 | 84.02 | 15.41 | 15.41 | 0.58 | 0.58 |
| validation | 2024-03 | 591,767 | 59,177 | 78.49 | 78.48 | 20.65 | 20.65 | 0.87 | 0.87 |
| validation | 2024-04 | 582,185 | 58,219 | 80.55 | 80.55 | 18.76 | 18.76 | 0.69 | 0.69 |
| validation | 2024-05 | 609,743 | 60,974 | 73.35 | 73.35 | 25.3 | 25.3 | 1.35 | 1.35 |
| validation | 2024-06 | 611,132 | 61,113 | 73.73 | 73.73 | 24.98 | 24.98 | 1.29 | 1.29 |
| validation | 2024-07 | 634,613 | 63,461 | 68.47 | 68.47 | 28.63 | 28.63 | 2.89 | 2.89 |
| validation | 2024-08 | 619,025 | 61,902 | 75.39 | 75.39 | 22.56 | 22.56 | 2.05 | 2.05 |
| validation | 2024-09 | 582,622 | 58,263 | 84.42 | 84.42 | 14.98 | 14.98 | 0.6 | 0.6 |
| validation | 2024-10 | 615,497 | 61,549 | 85.52 | 85.52 | 13.46 | 13.46 | 1.02 | 1.02 |
| validation | 2024-11 | 575,404 | 57,541 | 85.2 | 85.2 | 14.34 | 14.34 | 0.46 | 0.46 |
| validation | 2024-12 | 590,581 | 59,058 | 77.81 | 77.82 | 21.48 | 21.48 | 0.7 | 0.7 |

## 4. 기상 매칭

**출발 공항**

| split | matched | no_obs_within_2h | matched % |
|---|---|---|---|
| test | 699,382 | 779 | 99.889 |
| train | 5,153,143 | 3,625 | 99.93 |
| validation | 707,279 | 627 | 99.911 |

**도착 공항**

| split | matched | no_obs_within_2h | matched % |
|---|---|---|---|
| test | 699,554 | 607 | 99.913 |
| train | 5,153,739 | 3,029 | 99.941 |
| validation | 707,358 | 548 | 99.923 |

- 출발 공항 매칭 실패 편은 학습셋에서 빠지고 `unmatched_flights.parquet`에 보존했습니다. 도착 공항 매칭 실패 편은 학습셋에 남기고 `dest_*`를 NaN으로 둡니다.
- 매칭 실패로 학습셋에서 빠진 결항 편: 샘플 내 결항 289,146편 중 984편(0.34%).

### 매칭 실패가 많은 출발 공항 원인 조사

| Origin | iem_station | match_type | station_distance_km | failed | sampled_flights | failed_pct | median_gap_to_last_obs_min | top_local_dep_hours | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| BQN | TJBQ | faa_code | 0.0 | 1,030 | 3,188 | 32.31 | 222 | 02시:246, 03시:166, 06시:166, 01시:159 | 야간 정기 공백 |
| RSW | RSW | exact | 0.54 | 599 | 34,814 | 1.72 | 656 | 07시:67, 09시:50, 11시:49, 18시:45 | 수 시간 관측 공백 |
| PSE | TJPS | faa_code | 0.0 | 330 | 994 | 33.2 | 860 | 13시:49, 05시:48, 20시:40, 22시:36 | 수 시간 관측 공백 |
| FLL | FLL | exact | 1.17 | 266 | 96,680 | 0.28 | 662 | 07시:26, 19시:23, 20시:22, 08시:19 | 수 시간 관측 공백 |
| LGB | LGB | exact | 0.58 | 136 | 15,797 | 0.86 | 1032 | 07시:25, 13시:20, 08시:13, 09시:12 | 수 시간 관측 공백 |
| XNA | XNA | exact | 0.02 | 116 | 10,134 | 1.14 | 268 | 06시:46, 07시:32, 05시:23, 08시:9 | 수 시간 관측 공백 |
| AGS | AGS | exact | 0.0 | 107 | 3,173 | 3.37 | 6040 | 07시:20, 16시:16, 10시:11, 13시:10 | 하루 이상 관측 중단(장비 장애 추정) |
| HDN | HDN | exact | 0.22 | 87 | 2,465 | 3.53 | 1082 | 07시:28, 13시:28, 12시:17, 15시:5 | 수 시간 관측 공백 |
| CLT | CLT | exact | 1.4 | 86 | 186,942 | 0.05 | 153 | 20시:25, 16시:21, 17시:15, 11시:15 | 수 시간 관측 공백 |
| SBP | SBP | exact | 0.24 | 86 | 4,509 | 1.91 | 458 | 06시:43, 05시:16, 08시:13, 12시:6 | 수 시간 관측 공백 |

- `median_gap_to_last_obs_min`: 실패한 편의 예측 시점에서 직전 관측까지 간격(허용 시간 없이 찾은 값). 120분보다 크면 관측 공백입니다.
- 모든 상위 공항은 공항 구내 관측소(매핑 거리 3km 이내)에 정상 매핑되어 있습니다. 매핑 문제는 없었습니다.
- BQN 연도별 관측 커버리지: 2022:64.61%, 2023:65.81%, 2024:66.94%, 2025:70.39%
- RSW 연도별 관측 커버리지: 2022:96.2%, 2023:98.97%, 2024:99.12%, 2025:99.76%
- PSE 연도별 관측 커버리지: 2022:13.95%, 2023:88.05%, 2024:97.35%, 2025:96.1%
- FLL 연도별 관측 커버리지: 2022:99.95%, 2023:99.68%, 2024:99.95%, 2025:99.9%

**v1에서 실패율이 높았던 BQN·GCC (v2 기간)**

| Origin | failed | sampled_flights | failed_pct | median_gap_to_last_obs_min |
|---|---|---|---|---|
| BQN | 1,030 | 3,188 | 32.31 | 222 |
| GCC | 16 | 509 | 3.14 | 5220 |

- **BQN**: IEM 관측이 매일 현지 01~05시(UTC 05~09시)에 없습니다(연간 커버리지 65~70%). 이른 아침 출발 편은 예측 시점(3시간 전)이 이 공백에 걸려 실패합니다. 매핑 문제가 아니라 관측 공백이라 수정하지 않았습니다.
- **GCC**: v1(2025년 1~3월)의 22% 실패는 그 기간의 관측 중단 때문이었습니다. v2 전체 기간 기준 실패율은 위 표와 같습니다.
- 그 밖의 상위 공항(RSW, FLL, LGB, AGS 등)은 실패율이 낮지만, 직전 관측까지 간격이 수 시간~수일입니다. 특정 기간에 관측소 데이터가 끊긴 것으로, 매핑은 정상입니다. 모두 관측 공백으로 기록했습니다.

## 5. UTC 변환

| tz_status | flights |
|---|---|
| ok | 27,657,677 |
| ambiguous_dst_assumed | 27 |

- 예정 소요시간 교차검증에서 5분 넘게 어긋난 편: 1,191 / 27,657,704 (0.0043%). v1과 같이 서머타임 전환 주말 등 BTS 예정 도착 시각 기록 불일치로, 출발 UTC에는 영향이 없습니다.

## 6. 데이터 누수 검증

| check | violations |
|---|---|
| 출발 공항: 관측 시각 > 예측 시각 | 0 |
| 출발 공항: 관측이 예측 시각 − 2시간보다 오래됨 | 0 |
| 도착 공항: 관측 시각 > 예측 시각 | 0 |
| 도착 공항: 관측이 예측 시각 − 2시간보다 오래됨 | 0 |
| 예측 시각 ≠ 예정 출발 UTC − 3시간 (병합본 전체) | 0 |
| BTS 기반 피처가 쓴 데이터의 마지막 날 > 운항월 M−3월 말 (행 단위) | 0 |
|   └ 피처 그룹별 합계 (hist_* 6종, proxy_ly, tail_hist, proxy_recent/yoy) | 0 |
| v2와 같은 행(flight_id 집합)이 아닌 달 | 0 |
| v2와 label이 다른 행 | 0 |
| 학습 피처 목록에 사후 정보·*_oracle·샘플링 컬럼·label_arr 포함 | 0 |
| Train/Validation/Test 기간 겹침 | 0 |

- 기상 3시간 창 특징과 기온 spike QC는 과거 관측만 사용합니다(v1과 동일).
- 당일 계획 편수 `*_oracle`, v2의 `*_proxy_v2_unavailable`, 당일 기체 일정 `*_sameday`는 병합본에만 있고 학습 피처에서 뺐습니다.
- `tail_*_hist`는 과거 참조일의 예정 시각만 쓰고, 운항일 당일의 Tail_Number·실제 시각·지연은 쓰지 않습니다.

### 공개 시점 검사

| feature_group | rows_over_limit |
|---|---|
| hist_carrier | 0 |
| hist_origin | 0 |
| hist_dest | 0 |
| hist_route | 0 |
| hist_origin_hour | 0 |
| hist_carrier_origin | 0 |
| tail_hist | 0 |
| proxy_ly | 0 |
| proxy_recent/origin_yoy | 0 |

- 각 행의 `bts_source_max_date`(BTS 기반 피처가 쓴 마지막 날)가 `bts_source_limit_date`(M−3월 말)를 넘는 행: **0건**.
- **공개 시각 한계:** 기상 시각은 관측 시각이며 실제 배포 시각은 데이터에 없습니다. `OBS_PUBLISH_LAG_MIN`=0분.

## 7. 과거 실적(hist_*)이 BTS 전체로 계산됐는지

| table | flights_counted | bts_rows_expected | equal | missing_in_sample |
|---|---|---|---|---|
| hist_carrier | 34,388,825 | 34,388,825 | 1 | 0 |
| hist_origin | 34,388,825 | 34,388,825 | 1 | 312 |
| hist_dest | 34,388,825 | 34,388,825 | 1 | 325 |
| hist_route | 34,388,825 | 34,388,825 | 1 | 16,508 |
| hist_origin_hour | 34,388,825 | 34,388,825 | 1 | 11,906 |
| hist_carrier_origin | 34,388,825 | 34,388,825 | 1 | 8,605 |

- 집계표의 편수 합계가 2020~2025년 BTS 원본 행 수와 같으면, 샘플이 아니라 전체로 계산된 것입니다.
- `missing_in_sample`: 창(M−14~M−3)에 해당 그룹 실적이 없어 NaN인 샘플 행 수. 신규 노선·시간대가 해당합니다.
- **2020년 말 코로나 영향:** 2022년 1~2월 운항편의 창은 2020-11~2021-10, 2020-12~2021-11입니다. 2020년 말은 운항 감소·결항 증가 시기(2020년 결항률 6.0%)라 이 달들의 hist_*는 이후 기간보다 결항률이 높게 잡힐 수 있습니다. train 초반 몇 달에만 해당합니다.

## 8. 혼잡도 proxy vs oracle (피어슨 상관, 샘플 전 데이터셋 전체)

| feature | proxy_ly | proxy_recent | v2_proxy_unavailable |
|---|---|---|---|
| origin_sched_deps_hour | 0.931 | 0.957 | 0.985 |
| origin_sched_deps_day | 0.982 | 0.988 | 0.996 |
| dest_sched_arrs_hour | 0.936 | 0.958 | 0.985 |
| carrier_sched_deps_origin_day | 0.987 | 0.991 | 0.996 |

- 평균 상관: proxy_ly 0.9591, proxy_recent 0.9733 → **`proxy_recent`를 base_plus_candidates에** 넣고, 다른 쪽은 `alt_congestion_candidates`로 남겼습니다.
- v2 proxy(D−7~28)는 상관이 가장 높지만 서비스 시점에 그 BTS가 공개돼 있지 않아 쓸 수 없습니다(참고용).
- `origin_yoy_ratio`는 작년 대비 운항량 추세 보정용 보조 피처로 base_plus_candidates에 넣었습니다.

## 9. 기체 일정 피처

### v2 당일 Tail_Number 결측률 (label별, 샘플 전) — 누수 원인

| split | label | n | n_missing | missing % |
|---|---|---|---|---|
| test | 0 정상 | 5,399,588 | 0 | 0 |
| test | 1 지연(≥15분) | 1,499,155 | 0 | 0 |
| test | 2 결항 | 102,876 | 12,501 | 12.152 |
| train | 0 정상 | 10,525,320 | 0 | 0 |
| train | 1 지연(≥15분) | 2,782,475 | 0 | 0 |
| train | 2 결항 | 269,229 | 50,046 | 18.589 |
| validation | 0 정상 | 5,541,614 | 0 | 0 |
| validation | 1 지연(≥15분) | 1,441,132 | 0 | 0 |
| validation | 2 결항 | 96,315 | 20,152 | 20.923 |

- 결항 편의 당일 Tail_Number 결측률 17.65%, 결항 아닌 편 0.0%. v2의 당일 `tail_*`는 결측이 곧 결항이라 `*_sameday`로 이름을 바꿔 학습에서 뺐습니다.

### v3 tail_*_hist 매칭률 (참조일에 같은 편을 찾은 비율, 샘플 기준)

| split | label | n | n_found | weighted_n | weighted_found | found % | missing % (tail_leg_index_hist) |
|---|---|---|---|---|---|---|---|
| test | 0 정상 | 539,959 | 289,892 | 5,399,590 | 2,898,920 | 53.69 | 46.31 |
| test | 1 지연(≥15분) | 149,916 | 76,365 | 1,499,160 | 763,650 | 50.94 | 49.06 |
| test | 2 결항 | 10,286 | 5,730 | 102,860 | 57,300 | 55.71 | 44.29 |
| train | 0 정상 | 2,105,064 | 1,014,427 | 10,525,320 | 5,072,135 | 48.19 | 51.81 |
| train | 1 지연(≥15분) | 2,782,475 | 1,287,406 | 2,782,475 | 1,287,406 | 46.27 | 53.73 |
| train | 2 결항 | 269,229 | 125,288 | 269,229 | 125,288 | 46.54 | 53.46 |
| validation | 0 정상 | 554,161 | 271,332 | 5,541,610 | 2,713,320 | 48.96 | 51.04 |
| validation | 1 지연(≥15분) | 144,114 | 67,601 | 1,441,140 | 676,010 | 46.91 | 53.09 |
| validation | 2 결항 | 9,631 | 4,770 | 96,310 | 47,700 | 49.53 | 50.47 |

- 원래 분포 기준(가중) 매칭률: test 53.13%, train 47.76%, validation 48.55%. 편명이 해마다 바뀌어 매칭이 안 되는 편이 있습니다.
- label별 매칭률 차이가 작으면 결측이 결항을 알려주지 않는다는 뜻입니다. 결측 누수 검사 결과:
  - `tail_*_hist` 경고 없음 → **통과**

## 10. 학습 피처 목록 점검

| list | n_features | forbidden_found |
|---|---|---|
| base | 21 | 0건 |
| base_plus_candidates | 70 | 0건 |
| no_weather | 32 | 0건 |
| with_dest_weather | 94 | 0건 |
| experimental_tail | 99 | 0건 |

## 11. 학습셋 클래스 분포 (ml/, 기상 매칭 성공 행)

| split | label | rows | pct | weighted_pct |
|---|---|---|---|---|
| train | 0 정상 | 2,103,874 | 40.827 | 77.527 |
| train | 1 지연(≥15분) | 2,781,002 | 53.967 | 20.496 |
| train | 2 결항 | 268,267 | 5.206 | 1.977 |
| validation | 0 정상 | 553,650 | 78.279 | 78.279 |
| validation | 1 지연(≥15분) | 144,010 | 20.361 | 20.361 |
| validation | 2 결항 | 9,619 | 1.36 | 1.36 |
| test | 0 정상 | 539,316 | 77.113 | 77.113 |
| test | 1 지연(≥15분) | 149,790 | 21.417 | 21.417 |
| test | 2 결항 | 10,276 | 1.469 | 1.469 |

## 12. 기상 원본 QC (v2 재사용)

| column | values_set_to_nan |
|---|---|
| duplicates_removed | 2,534 |
| tmpf | 9 |
| dwpf | 10 |
| relh | 0 |
| drct | 0 |
| sknt | 42 |
| gust | 36 |
| peak_wind_gust | 0 |
| vsby | 39 |
| alti | 211 |
| mslp | 386 |
| p01i | 10 |
| skyl1 | 995 |
| skyl2 | 1,297 |
| skyl3 | 917 |
| skyl4 | 394 |
| tmpf_spike | 2,476 |
| dwpf_spike | 2,983 |

## 13. 학습 피처 결측률 (split별 %)

| feature | lists | missing_% train | missing_% validation | missing_% test |
|---|---|---|---|---|
| Reporting_Airline | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| Origin | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| Dest | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| dep_hour_local | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| day_of_week | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| is_weekend | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| month | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| season | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| Distance | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| tmpf | base,base_plus_candidates,with_dest_weather,experimental_tail | 0.04 | 0.142 | 0.038 |
| sknt | base,base_plus_candidates,with_dest_weather,experimental_tail | 0.15 | 0.147 | 0.119 |
| wind_max_kt | base,base_plus_candidates,with_dest_weather,experimental_tail | 0.149 | 0.146 | 0.118 |
| vsby | base,base_plus_candidates,with_dest_weather,experimental_tail | 0.046 | 0.061 | 0.044 |
| ceiling_ft | base,base_plus_candidates,with_dest_weather,experimental_tail | 0.04 | 0.076 | 0.033 |
| flight_category | base,base_plus_candidates,with_dest_weather,experimental_tail | 0.065 | 0.101 | 0.065 |
| p01i | base,base_plus_candidates,with_dest_weather,experimental_tail | 0.034 | 0.036 | 0.003 |
| wx_thunder | base,base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| wx_snow | base,base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| wx_rain | base,base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| wx_freezing | base,base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| wx_fog | base,base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| route | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| arr_hour_local | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| CRSElapsedTime | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| overnight_arrival | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| day_of_month | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| days_to_federal_holiday | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| origin_sched_deps_hour_proxy_recent | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| origin_sched_deps_day_proxy_recent | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| dest_sched_arrs_hour_proxy_recent | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| carrier_sched_deps_origin_day_proxy_recent | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| origin_yoy_ratio | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 4.224 | 0.066 | 0.124 |
| hist_carrier_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| hist_carrier_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| hist_origin_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.003 | 0.004 | 0.02 |
| hist_origin_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.003 | 0.004 | 0.02 |
| hist_dest_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.003 | 0.007 | 0.018 |
| hist_dest_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.003 | 0.007 | 0.018 |
| hist_route_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.244 | 0.251 | 0.306 |
| hist_route_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.244 | 0.251 | 0.306 |
| hist_origin_hour_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.185 | 0.131 | 0.169 |
| hist_origin_hour_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.185 | 0.131 | 0.169 |
| hist_carrier_origin_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.139 | 0.091 | 0.104 |
| hist_carrier_origin_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | 0.139 | 0.091 | 0.104 |
| dwpf | base_plus_candidates,with_dest_weather,experimental_tail | 0.069 | 0.182 | 0.072 |
| relh | base_plus_candidates,with_dest_weather,experimental_tail | 0.051 | 0.172 | 0.045 |
| temp_dew_spread_f | base_plus_candidates,with_dest_weather,experimental_tail | 0.075 | 0.185 | 0.078 |
| wind_dir_sin | base_plus_candidates,with_dest_weather,experimental_tail | 4.243 | 4.052 | 3.956 |
| wind_dir_cos | base_plus_candidates,with_dest_weather,experimental_tail | 4.243 | 4.052 | 3.956 |
| gust_reported | base_plus_candidates,with_dest_weather,experimental_tail | 0.15 | 0.147 | 0.119 |
| strong_wind | base_plus_candidates,with_dest_weather,experimental_tail | 0.15 | 0.147 | 0.119 |
| low_ceiling | base_plus_candidates,with_dest_weather,experimental_tail | 0.04 | 0.076 | 0.033 |
| low_visibility | base_plus_candidates,with_dest_weather,experimental_tail | 0.046 | 0.061 | 0.044 |
| wx_hail | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| wx_mist_haze | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| wx_heavy | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| precip_any | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| icing_risk | base_plus_candidates,with_dest_weather,experimental_tail | 0.04 | 0.142 | 0.038 |
| weather_severity | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| alti | base_plus_candidates,with_dest_weather,experimental_tail | 0.015 | 0.033 | 0.023 |
| thunder_3h | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| snow_3h | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| freezing_3h | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| max_wind_3h_kt | base_plus_candidates,with_dest_weather,experimental_tail | 0.018 | 0.03 | 0.014 |
| min_vsby_3h | base_plus_candidates,with_dest_weather,experimental_tail | 0.024 | 0.029 | 0.016 |
| min_ceiling_3h_ft | base_plus_candidates,with_dest_weather,experimental_tail | 0.015 | 0.03 | 0.011 |
| precip_3h_in | base_plus_candidates,with_dest_weather,experimental_tail | 0.179 | 0.144 | 0.341 |
| n_special_obs_3h | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| alti_change_3h | base_plus_candidates,with_dest_weather,experimental_tail | 1.88 | 1.685 | 1.788 |
| obs_age_min | base_plus_candidates,with_dest_weather,experimental_tail | 0 | 0 | 0 |
| dest_ceiling_ft | with_dest_weather,experimental_tail | 0.097 | 0.168 | 0.123 |
| dest_vsby | with_dest_weather,experimental_tail | 0.101 | 0.127 | 0.125 |
| dest_flight_category | with_dest_weather,experimental_tail | 0.121 | 0.188 | 0.149 |
| dest_wind_max_kt | with_dest_weather,experimental_tail | 0.212 | 0.224 | 0.203 |
| dest_wx_thunder | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_wx_snow | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_wx_rain | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_wx_freezing | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_wx_hail | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_wx_fog | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_wx_mist_haze | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_wx_heavy | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_precip_any | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_weather_severity | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_thunder_3h | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_snow_3h | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_freezing_3h | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_max_wind_3h_kt | with_dest_weather,experimental_tail | 0.075 | 0.109 | 0.101 |
| dest_min_vsby_3h | with_dest_weather,experimental_tail | 0.079 | 0.104 | 0.102 |
| dest_min_ceiling_3h_ft | with_dest_weather,experimental_tail | 0.07 | 0.13 | 0.1 |
| dest_precip_3h_in | with_dest_weather,experimental_tail | 0.214 | 0.179 | 0.396 |
| dest_n_special_obs_3h | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| dest_alti_change_3h | with_dest_weather,experimental_tail | 1.863 | 1.764 | 1.853 |
| dest_obs_age_min | with_dest_weather,experimental_tail | 0.059 | 0.073 | 0.086 |
| tail_leg_index_hist | experimental_tail | 52.943 | 51.445 | 46.872 |
| tail_sched_turn_min_hist | experimental_tail | 60.526 | 60.71 | 57.16 |
| tail_prev_origin_hist | experimental_tail | 60.526 | 60.71 | 57.16 |
| tail_prev_sched_elapsed_hist | experimental_tail | 60.526 | 60.71 | 57.16 |
| tail_ref_days_found | experimental_tail | 0 | 0 | 0 |

## 14. 기존 processed/ (v1) 대비 변경점

| 항목 | v1 | v2 |
|---|---|---|
| 기간·분할 | 2025-01~03, 월 단위 1/2/3월 | train 2022~23 / val 2024 / test 2025 (각 12개월) |
| 샘플링 | 없음 (164만 편 전체) | train 정상만 0.2, val·test 0.1 균등, sample_rate/weight |
| 행 수 (학습셋) | 1,644,539 | 6,559,804 |
| hist_* | 2024년 고정 1년 | 운항월 직전 12개월 창(월 단위), BTS 전체로 계산 |
| 혼잡도 | 당일 계획 편수 (서비스 시 알 수 없음) | *_proxy(직전 4주 같은 요일 평균), 당일 값은 *_oracle로 학습 제외 |
| 도착 공항 기상 | 없음 | dest_* 24개 (같은 예측 시점 기준) |
| 기체 일정 | 없음 | tail_* 4개 (experimental_tail 전용) |
| month·season | Train 내 상수라 제외 | base에 포함 |
| 도착 기준 레이블 | 없음 | label_arr (병합본, 평가용) |
| 피처 목록 | base, base_plus_candidates | + no_weather, with_dest_weather, experimental_tail |
| 포맷 | CSV + pickle | Parquet(snappy)만 |
| 최대 메모리 | 측정 안 함 (전체 적재) | 3,429 MB |

## 15. 결측 누수 검사 (모든 학습 피처, 가중 = 원래 분포 기준)

결측 행과 값 있는 행의 label 분포를 split별로 비교했습니다. 경고 기준: 결측 행이 100행 이상이면서 (a) 결측 행 결항률 ≥ 전체 결항률×3 또는 (b) 결측 행 지연률이 전체와 10%p 이상 차이.

### 경고 목록과 판단

| split | feature | missing_rows | missing_% | 결측: 정상/지연/결항 % | 전체 결항 % | warning | 원인 | 누수 판단 |
|---|---|---|---|---|---|---|---|---|
| train | tmpf | 2,068 | 0.04 | 70.42/19.24/10.35 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | vsby | 2,350 | 0.046 | 73.08/17.67/9.24 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | ceiling_ft | 2,055 | 0.04 | 71.34/18.65/10.01 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | flight_category | 3,370 | 0.065 | 74.50/17.93/7.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | origin_yoy_ratio | 217,678 | 4.224 | 75.50/18.35/6.14 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 1년 전 같은 달(M−15월)에 운항 실적이 없던 공항(신규·재취항, 특히 2020~21년 코로나 감편). 공개된 BTS로 계산되므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | hist_origin_delay_rate | 145 | 0.003 | 75.34/15.07/9.59 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 직전 12개월(공개분)에 그 그룹 실적이 없는 신규 노선·시간대·공항. 신규 편은 지연·결항 양상이 다를 수 있으나 예측 시점에 알 수 있는 정보 | 누수 아님 → 유지 |
| train | hist_origin_cancel_rate | 145 | 0.003 | 75.34/15.07/9.59 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 직전 12개월(공개분)에 그 그룹 실적이 없는 신규 노선·시간대·공항. 신규 편은 지연·결항 양상이 다를 수 있으나 예측 시점에 알 수 있는 정보 | 누수 아님 → 유지 |
| train | hist_dest_delay_rate | 151 | 0.003 | 69.97/18.95/11.08 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 직전 12개월(공개분)에 그 그룹 실적이 없는 신규 노선·시간대·공항. 신규 편은 지연·결항 양상이 다를 수 있으나 예측 시점에 알 수 있는 정보 | 누수 아님 → 유지 |
| train | hist_dest_cancel_rate | 151 | 0.003 | 69.97/18.95/11.08 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 직전 12개월(공개분)에 그 그룹 실적이 없는 신규 노선·시간대·공항. 신규 편은 지연·결항 양상이 다를 수 있으나 예측 시점에 알 수 있는 정보 | 누수 아님 → 유지 |
| train | dwpf | 3,561 | 0.069 | 70.11/22.12/7.77 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | relh | 2,619 | 0.051 | 73.90/17.59/8.51 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | temp_dew_spread_f | 3,882 | 0.075 | 70.02/22.42/7.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | low_ceiling | 2,055 | 0.04 | 71.34/18.65/10.01 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | low_visibility | 2,350 | 0.046 | 73.08/17.67/9.24 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | icing_risk | 2,068 | 0.04 | 70.42/19.24/10.35 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | max_wind_3h_kt | 923 | 0.018 | 66.58/14.43/18.99 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 3시간 창 안에 정시 관측이 없거나 3시간 전 기압 관측이 없음(관측 간격 큼) | 누수 아님 → 유지 |
| train | min_vsby_3h | 1,230 | 0.024 | 70.21/15.00/14.79 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 3시간 창 안에 정시 관측이 없거나 3시간 전 기압 관측이 없음(관측 간격 큼) | 누수 아님 → 유지 |
| train | min_ceiling_3h_ft | 752 | 0.015 | 61.49/15.95/22.57 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 3시간 창 안에 정시 관측이 없거나 3시간 전 기압 관측이 없음(관측 간격 큼) | 누수 아님 → 유지 |
| train | dest_ceiling_ft | 4,976 | 0.097 | 70.72/18.67/10.60 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_vsby | 5,189 | 0.101 | 71.43/17.97/10.60 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_flight_category | 6,221 | 0.121 | 72.52/18.34/9.14 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_wx_thunder | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_wx_snow | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_wx_rain | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_wx_freezing | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_wx_hail | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_wx_fog | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_wx_mist_haze | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_wx_heavy | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_precip_any | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_weather_severity | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_thunder_3h | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_snow_3h | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_freezing_3h | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_max_wind_3h_kt | 3,866 | 0.075 | 68.98/17.84/13.18 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_min_vsby_3h | 4,057 | 0.079 | 69.84/17.43/12.74 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_min_ceiling_3h_ft | 3,625 | 0.07 | 67.77/18.26/13.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_n_special_obs_3h | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| train | dest_obs_age_min | 3,019 | 0.059 | 68.90/18.54/12.56 | 1.98 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | tmpf | 1,004 | 0.142 | 74.10/18.33/7.57 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | sknt | 1,039 | 0.147 | 75.55/19.54/4.91 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | wind_max_kt | 1,033 | 0.146 | 75.51/19.55/4.94 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | vsby | 428 | 0.061 | 77.80/14.25/7.94 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | ceiling_ft | 540 | 0.076 | 80.37/12.59/7.04 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | flight_category | 712 | 0.101 | 79.49/14.04/6.46 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | dwpf | 1,288 | 0.182 | 76.63/16.54/6.83 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | relh | 1,215 | 0.172 | 76.30/16.71/7.00 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | temp_dew_spread_f | 1,311 | 0.185 | 76.43/16.86/6.71 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | gust_reported | 1,039 | 0.147 | 75.55/19.54/4.91 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | strong_wind | 1,039 | 0.147 | 75.55/19.54/4.91 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | low_ceiling | 540 | 0.076 | 80.37/12.59/7.04 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | low_visibility | 428 | 0.061 | 77.80/14.25/7.94 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | icing_risk | 1,004 | 0.142 | 74.10/18.33/7.57 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | max_wind_3h_kt | 214 | 0.03 | 73.36/11.21/15.42 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 3시간 창 안에 정시 관측이 없거나 3시간 전 기압 관측이 없음(관측 간격 큼) | 누수 아님 → 유지 |
| validation | min_vsby_3h | 205 | 0.029 | 75.12/13.66/11.22 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 3시간 창 안에 정시 관측이 없거나 3시간 전 기압 관측이 없음(관측 간격 큼) | 누수 아님 → 유지 |
| validation | min_ceiling_3h_ft | 214 | 0.03 | 76.17/14.02/9.81 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 3시간 창 안에 정시 관측이 없거나 3시간 전 기압 관측이 없음(관측 간격 큼) | 누수 아님 → 유지 |
| validation | dest_ceiling_ft | 1,188 | 0.168 | 79.88/15.74/4.38 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | dest_vsby | 895 | 0.127 | 79.22/16.20/4.58 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | dest_flight_category | 1,328 | 0.188 | 79.97/15.89/4.14 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | dest_max_wind_3h_kt | 774 | 0.109 | 80.49/14.60/4.91 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| validation | dest_min_vsby_3h | 739 | 0.104 | 80.24/15.29/4.47 | 1.36 | 결측 행 결항률 ≥ 전체×3 | 도착 공항 관측 공백(관측소 장애·야간 미관측). 악천후 중 관측 중단과 겹쳐 결항률이 높게 나올 수 있음. 예측 시점에도 같은 공백이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| test | vsby | 311 | 0.044 | 75.88/19.61/4.50 | 1.47 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| test | relh | 317 | 0.045 | 73.82/21.77/4.42 | 1.47 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| test | low_visibility | 311 | 0.044 | 75.88/19.61/4.50 | 1.47 | 결측 행 결항률 ≥ 전체×3 | 출발 공항 관측값 일부 결측(센서·보고 누락). 결측 행의 결항률이 높은 것은 악천후 중 센서 장애와 겹쳤을 가능성. 예측 시점의 최신 관측에서도 같은 결측이 보이므로 사후 정보 아님 | 누수 아님 → 유지 |
| test | min_vsby_3h | 109 | 0.016 | 73.39/18.35/8.26 | 1.47 | 결측 행 결항률 ≥ 전체×3 | 3시간 창 안에 정시 관측이 없거나 3시간 전 기압 관측이 없음(관측 간격 큼) | 누수 아님 → 유지 |

### 전체 결과표

| split | feature | missing_rows | missing_% | 결측: 정상/지연/결항 % | 값 있음: 정상/지연/결항 % | 전체 결항 % | warning |
|---|---|---|---|---|---|---|---|
| train | tmpf | 2,068 | 0.04 | 70.42/19.24/10.35 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | sknt | 7,719 | 0.15 | 72.25/23.57/4.18 | 77.53/20.49/1.97 | 1.98 |  |
| train | wind_max_kt | 7,685 | 0.149 | 72.33/23.49/4.18 | 77.53/20.49/1.97 | 1.98 |  |
| train | vsby | 2,350 | 0.046 | 73.08/17.67/9.24 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | ceiling_ft | 2,055 | 0.04 | 71.34/18.65/10.01 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | flight_category | 3,370 | 0.065 | 74.50/17.93/7.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | p01i | 1,768 | 0.034 | 73.32/23.71/2.97 | 77.53/20.49/1.98 | 1.98 |  |
| train | CRSElapsedTime | 4 | 0 | 0.00/0.00/100.00 | 77.53/20.50/1.98 | 1.98 |  |
| train | origin_yoy_ratio | 217,678 | 4.224 | 75.50/18.35/6.14 | 77.61/20.59/1.80 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | hist_origin_delay_rate | 145 | 0.003 | 75.34/15.07/9.59 | 77.53/20.50/1.98 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | hist_origin_cancel_rate | 145 | 0.003 | 75.34/15.07/9.59 | 77.53/20.50/1.98 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | hist_dest_delay_rate | 151 | 0.003 | 69.97/18.95/11.08 | 77.53/20.50/1.98 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | hist_dest_cancel_rate | 151 | 0.003 | 69.97/18.95/11.08 | 77.53/20.50/1.98 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | hist_route_delay_rate | 12,563 | 0.244 | 76.14/20.40/3.46 | 77.53/20.50/1.97 | 1.98 |  |
| train | hist_route_cancel_rate | 12,563 | 0.244 | 76.14/20.40/3.46 | 77.53/20.50/1.97 | 1.98 |  |
| train | hist_origin_hour_delay_rate | 9,523 | 0.185 | 81.10/15.66/3.24 | 77.52/20.51/1.97 | 1.98 |  |
| train | hist_origin_hour_cancel_rate | 9,523 | 0.185 | 81.10/15.66/3.24 | 77.52/20.51/1.97 | 1.98 |  |
| train | hist_carrier_origin_delay_rate | 7,168 | 0.139 | 76.72/20.85/2.42 | 77.53/20.50/1.98 | 1.98 |  |
| train | hist_carrier_origin_cancel_rate | 7,168 | 0.139 | 76.72/20.85/2.42 | 77.53/20.50/1.98 | 1.98 |  |
| train | dwpf | 3,561 | 0.069 | 70.11/22.12/7.77 | 77.53/20.49/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | relh | 2,619 | 0.051 | 73.90/17.59/8.51 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | temp_dew_spread_f | 3,882 | 0.075 | 70.02/22.42/7.56 | 77.53/20.49/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | wind_dir_sin | 218,662 | 4.243 | 73.86/24.50/1.64 | 77.68/20.33/1.99 | 1.98 |  |
| train | wind_dir_cos | 218,662 | 4.243 | 73.86/24.50/1.64 | 77.68/20.33/1.99 | 1.98 |  |
| train | gust_reported | 7,719 | 0.15 | 72.25/23.57/4.18 | 77.53/20.49/1.97 | 1.98 |  |
| train | strong_wind | 7,719 | 0.15 | 72.25/23.57/4.18 | 77.53/20.49/1.97 | 1.98 |  |
| train | low_ceiling | 2,055 | 0.04 | 71.34/18.65/10.01 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | low_visibility | 2,350 | 0.046 | 73.08/17.67/9.24 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | icing_risk | 2,068 | 0.04 | 70.42/19.24/10.35 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | alti | 774 | 0.015 | 77.62/18.22/4.16 | 77.53/20.50/1.98 | 1.98 |  |
| train | max_wind_3h_kt | 923 | 0.018 | 66.58/14.43/18.99 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | min_vsby_3h | 1,230 | 0.024 | 70.21/15.00/14.79 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | min_ceiling_3h_ft | 752 | 0.015 | 61.49/15.95/22.57 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | precip_3h_in | 9,248 | 0.179 | 77.95/19.53/2.52 | 77.53/20.50/1.98 | 1.98 |  |
| train | alti_change_3h | 96,884 | 1.88 | 70.37/25.77/3.87 | 77.65/20.41/1.95 | 1.98 |  |
| train | dest_ceiling_ft | 4,976 | 0.097 | 70.72/18.67/10.60 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_vsby | 5,189 | 0.101 | 71.43/17.97/10.60 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_flight_category | 6,221 | 0.121 | 72.52/18.34/9.14 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_wind_max_kt | 10,910 | 0.212 | 71.73/22.37/5.90 | 77.54/20.49/1.97 | 1.98 |  |
| train | dest_wx_thunder | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_wx_snow | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_wx_rain | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_wx_freezing | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_wx_hail | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_wx_fog | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_wx_mist_haze | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_wx_heavy | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_precip_any | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_weather_severity | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_thunder_3h | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_snow_3h | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_freezing_3h | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_max_wind_3h_kt | 3,866 | 0.075 | 68.98/17.84/13.18 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_min_vsby_3h | 4,057 | 0.079 | 69.84/17.43/12.74 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_min_ceiling_3h_ft | 3,625 | 0.07 | 67.77/18.26/13.97 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_precip_3h_in | 11,050 | 0.214 | 73.23/21.59/5.18 | 77.54/20.49/1.97 | 1.98 |  |
| train | dest_n_special_obs_3h | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | dest_alti_change_3h | 96,024 | 1.863 | 72.26/23.87/3.87 | 77.62/20.44/1.94 | 1.98 |  |
| train | dest_obs_age_min | 3,019 | 0.059 | 68.90/18.54/12.56 | 77.53/20.50/1.97 | 1.98 | 결측 행 결항률 ≥ 전체×3 |
| train | tail_leg_index_hist | 2,728,225 | 52.943 | 76.89/21.08/2.02 | 78.22/19.85/1.93 | 1.98 |  |
| train | tail_sched_turn_min_hist | 3,119,016 | 60.526 | 78.25/19.74/2.01 | 76.37/21.71/1.93 | 1.98 |  |
| train | tail_prev_origin_hist | 3,119,016 | 60.526 | 78.25/19.74/2.01 | 76.37/21.71/1.93 | 1.98 |  |
| train | tail_prev_sched_elapsed_hist | 3,119,016 | 60.526 | 78.25/19.74/2.01 | 76.37/21.71/1.93 | 1.98 |  |
| validation | tmpf | 1,004 | 0.142 | 74.10/18.33/7.57 | 78.28/20.36/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | sknt | 1,039 | 0.147 | 75.55/19.54/4.91 | 78.28/20.36/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | wind_max_kt | 1,033 | 0.146 | 75.51/19.55/4.94 | 78.28/20.36/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | vsby | 428 | 0.061 | 77.80/14.25/7.94 | 78.28/20.36/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | ceiling_ft | 540 | 0.076 | 80.37/12.59/7.04 | 78.28/20.37/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | flight_category | 712 | 0.101 | 79.49/14.04/6.46 | 78.28/20.37/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | p01i | 255 | 0.036 | 72.16/25.49/2.35 | 78.28/20.36/1.36 | 1.36 |  |
| validation | origin_yoy_ratio | 470 | 0.066 | 78.30/18.94/2.77 | 78.28/20.36/1.36 | 1.36 |  |
| validation | hist_origin_delay_rate | 26 | 0.004 | 57.69/30.77/11.54 | 78.28/20.36/1.36 | 1.36 |  |
| validation | hist_origin_cancel_rate | 26 | 0.004 | 57.69/30.77/11.54 | 78.28/20.36/1.36 | 1.36 |  |
| validation | hist_dest_delay_rate | 48 | 0.007 | 72.92/27.08/0.00 | 78.28/20.36/1.36 | 1.36 |  |
| validation | hist_dest_cancel_rate | 48 | 0.007 | 72.92/27.08/0.00 | 78.28/20.36/1.36 | 1.36 |  |
| validation | hist_route_delay_rate | 1,775 | 0.251 | 70.99/26.82/2.20 | 78.30/20.34/1.36 | 1.36 |  |
| validation | hist_route_cancel_rate | 1,775 | 0.251 | 70.99/26.82/2.20 | 78.30/20.34/1.36 | 1.36 |  |
| validation | hist_origin_hour_delay_rate | 926 | 0.131 | 79.59/18.57/1.84 | 78.28/20.36/1.36 | 1.36 |  |
| validation | hist_origin_hour_cancel_rate | 926 | 0.131 | 79.59/18.57/1.84 | 78.28/20.36/1.36 | 1.36 |  |
| validation | hist_carrier_origin_delay_rate | 646 | 0.091 | 70.43/26.93/2.63 | 78.29/20.36/1.36 | 1.36 |  |
| validation | hist_carrier_origin_cancel_rate | 646 | 0.091 | 70.43/26.93/2.63 | 78.29/20.36/1.36 | 1.36 |  |
| validation | dwpf | 1,288 | 0.182 | 76.63/16.54/6.83 | 78.28/20.37/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | relh | 1,215 | 0.172 | 76.30/16.71/7.00 | 78.28/20.37/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | temp_dew_spread_f | 1,311 | 0.185 | 76.43/16.86/6.71 | 78.28/20.37/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | wind_dir_sin | 28,658 | 4.052 | 74.34/24.38/1.28 | 78.45/20.19/1.36 | 1.36 |  |
| validation | wind_dir_cos | 28,658 | 4.052 | 74.34/24.38/1.28 | 78.45/20.19/1.36 | 1.36 |  |
| validation | gust_reported | 1,039 | 0.147 | 75.55/19.54/4.91 | 78.28/20.36/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | strong_wind | 1,039 | 0.147 | 75.55/19.54/4.91 | 78.28/20.36/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | low_ceiling | 540 | 0.076 | 80.37/12.59/7.04 | 78.28/20.37/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | low_visibility | 428 | 0.061 | 77.80/14.25/7.94 | 78.28/20.36/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | icing_risk | 1,004 | 0.142 | 74.10/18.33/7.57 | 78.28/20.36/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | alti | 236 | 0.033 | 85.17/12.29/2.54 | 78.28/20.36/1.36 | 1.36 |  |
| validation | max_wind_3h_kt | 214 | 0.03 | 73.36/11.21/15.42 | 78.28/20.36/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | min_vsby_3h | 205 | 0.029 | 75.12/13.66/11.22 | 78.28/20.36/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | min_ceiling_3h_ft | 214 | 0.03 | 76.17/14.02/9.81 | 78.28/20.36/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | precip_3h_in | 1,015 | 0.144 | 78.72/18.13/3.15 | 78.28/20.36/1.36 | 1.36 |  |
| validation | alti_change_3h | 11,921 | 1.685 | 71.12/26.21/2.68 | 78.40/20.26/1.34 | 1.36 |  |
| validation | dest_ceiling_ft | 1,188 | 0.168 | 79.88/15.74/4.38 | 78.28/20.37/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | dest_vsby | 895 | 0.127 | 79.22/16.20/4.58 | 78.28/20.37/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | dest_flight_category | 1,328 | 0.188 | 79.97/15.89/4.14 | 78.28/20.37/1.35 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | dest_wind_max_kt | 1,584 | 0.224 | 77.84/18.69/3.47 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_wx_thunder | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_wx_snow | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_wx_rain | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_wx_freezing | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_wx_hail | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_wx_fog | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_wx_mist_haze | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_wx_heavy | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_precip_any | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_weather_severity | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_thunder_3h | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_snow_3h | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_freezing_3h | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_max_wind_3h_kt | 774 | 0.109 | 80.49/14.60/4.91 | 78.28/20.37/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | dest_min_vsby_3h | 739 | 0.104 | 80.24/15.29/4.47 | 78.28/20.37/1.36 | 1.36 | 결측 행 결항률 ≥ 전체×3 |
| validation | dest_min_ceiling_3h_ft | 918 | 0.13 | 82.03/14.60/3.38 | 78.27/20.37/1.36 | 1.36 |  |
| validation | dest_precip_3h_in | 1,266 | 0.179 | 76.94/20.46/2.61 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_n_special_obs_3h | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | dest_alti_change_3h | 12,474 | 1.764 | 73.84/23.56/2.60 | 78.36/20.30/1.34 | 1.36 |  |
| validation | dest_obs_age_min | 518 | 0.073 | 80.50/16.60/2.90 | 78.28/20.36/1.36 | 1.36 |  |
| validation | tail_leg_index_hist | 363,859 | 51.445 | 77.65/21.01/1.33 | 78.94/19.67/1.39 | 1.36 |  |
| validation | tail_sched_turn_min_hist | 429,390 | 60.71 | 79.07/19.60/1.33 | 77.06/21.54/1.40 | 1.36 |  |
| validation | tail_prev_origin_hist | 429,390 | 60.71 | 79.07/19.60/1.33 | 77.06/21.54/1.40 | 1.36 |  |
| validation | tail_prev_sched_elapsed_hist | 429,390 | 60.71 | 79.07/19.60/1.33 | 77.06/21.54/1.40 | 1.36 |  |
| test | tmpf | 263 | 0.038 | 74.14/21.67/4.18 | 77.11/21.42/1.47 | 1.47 |  |
| test | sknt | 832 | 0.119 | 72.48/25.24/2.28 | 77.12/21.41/1.47 | 1.47 |  |
| test | wind_max_kt | 827 | 0.118 | 72.55/25.15/2.30 | 77.12/21.41/1.47 | 1.47 |  |
| test | vsby | 311 | 0.044 | 75.88/19.61/4.50 | 77.11/21.42/1.47 | 1.47 | 결측 행 결항률 ≥ 전체×3 |
| test | ceiling_ft | 234 | 0.033 | 75.64/20.51/3.85 | 77.11/21.42/1.47 | 1.47 |  |
| test | flight_category | 457 | 0.065 | 77.24/19.47/3.28 | 77.11/21.42/1.47 | 1.47 |  |
| test | p01i | 24 | 0.003 | 70.83/25.00/4.17 | 77.11/21.42/1.47 | 1.47 |  |
| test | CRSElapsedTime | 1 | 0 | 0.00/0.00/100.00 | 77.11/21.42/1.47 | 1.47 |  |
| test | origin_yoy_ratio | 864 | 0.124 | 76.39/20.83/2.78 | 77.11/21.42/1.47 | 1.47 |  |
| test | hist_origin_delay_rate | 139 | 0.02 | 77.70/18.71/3.60 | 77.11/21.42/1.47 | 1.47 |  |
| test | hist_origin_cancel_rate | 139 | 0.02 | 77.70/18.71/3.60 | 77.11/21.42/1.47 | 1.47 |  |
| test | hist_dest_delay_rate | 126 | 0.018 | 72.22/27.78/0.00 | 77.11/21.42/1.47 | 1.47 |  |
| test | hist_dest_cancel_rate | 126 | 0.018 | 72.22/27.78/0.00 | 77.11/21.42/1.47 | 1.47 |  |
| test | hist_route_delay_rate | 2,143 | 0.306 | 76.34/22.12/1.54 | 77.12/21.42/1.47 | 1.47 |  |
| test | hist_route_cancel_rate | 2,143 | 0.306 | 76.34/22.12/1.54 | 77.12/21.42/1.47 | 1.47 |  |
| test | hist_origin_hour_delay_rate | 1,182 | 0.169 | 75.89/22.00/2.12 | 77.12/21.42/1.47 | 1.47 |  |
| test | hist_origin_hour_cancel_rate | 1,182 | 0.169 | 75.89/22.00/2.12 | 77.12/21.42/1.47 | 1.47 |  |
| test | hist_carrier_origin_delay_rate | 726 | 0.104 | 74.24/23.83/1.93 | 77.12/21.41/1.47 | 1.47 |  |
| test | hist_carrier_origin_cancel_rate | 726 | 0.104 | 74.24/23.83/1.93 | 77.12/21.41/1.47 | 1.47 |  |
| test | dwpf | 502 | 0.072 | 72.11/24.10/3.78 | 77.12/21.42/1.47 | 1.47 |  |
| test | relh | 317 | 0.045 | 73.82/21.77/4.42 | 77.11/21.42/1.47 | 1.47 | 결측 행 결항률 ≥ 전체×3 |
| test | temp_dew_spread_f | 547 | 0.078 | 72.21/24.31/3.47 | 77.12/21.42/1.47 | 1.47 |  |
| test | wind_dir_sin | 27,665 | 3.956 | 72.49/26.31/1.19 | 77.30/21.22/1.48 | 1.47 |  |
| test | wind_dir_cos | 27,665 | 3.956 | 72.49/26.31/1.19 | 77.30/21.22/1.48 | 1.47 |  |
| test | gust_reported | 832 | 0.119 | 72.48/25.24/2.28 | 77.12/21.41/1.47 | 1.47 |  |
| test | strong_wind | 832 | 0.119 | 72.48/25.24/2.28 | 77.12/21.41/1.47 | 1.47 |  |
| test | low_ceiling | 234 | 0.033 | 75.64/20.51/3.85 | 77.11/21.42/1.47 | 1.47 |  |
| test | low_visibility | 311 | 0.044 | 75.88/19.61/4.50 | 77.11/21.42/1.47 | 1.47 | 결측 행 결항률 ≥ 전체×3 |
| test | icing_risk | 263 | 0.038 | 74.14/21.67/4.18 | 77.11/21.42/1.47 | 1.47 |  |
| test | alti | 160 | 0.023 | 84.38/13.12/2.50 | 77.11/21.42/1.47 | 1.47 |  |
| test | max_wind_3h_kt | 98 | 0.014 | 74.49/18.37/7.14 | 77.11/21.42/1.47 | 1.47 |  |
| test | min_vsby_3h | 109 | 0.016 | 73.39/18.35/8.26 | 77.11/21.42/1.47 | 1.47 | 결측 행 결항률 ≥ 전체×3 |
| test | min_ceiling_3h_ft | 77 | 0.011 | 76.62/15.58/7.79 | 77.11/21.42/1.47 | 1.47 |  |
| test | precip_3h_in | 2,382 | 0.341 | 77.08/21.75/1.18 | 77.11/21.42/1.47 | 1.47 |  |
| test | alti_change_3h | 12,502 | 1.788 | 70.05/27.01/2.94 | 77.24/21.32/1.44 | 1.47 |  |
| test | dest_ceiling_ft | 862 | 0.123 | 79.23/18.33/2.44 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_vsby | 874 | 0.125 | 79.06/18.42/2.52 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_flight_category | 1,044 | 0.149 | 79.41/18.20/2.39 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_wind_max_kt | 1,417 | 0.203 | 75.30/22.79/1.91 | 77.12/21.41/1.47 | 1.47 |  |
| test | dest_wx_thunder | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_wx_snow | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_wx_rain | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_wx_freezing | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_wx_hail | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_wx_fog | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_wx_mist_haze | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_wx_heavy | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_precip_any | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_weather_severity | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_thunder_3h | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_snow_3h | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_freezing_3h | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_max_wind_3h_kt | 708 | 0.101 | 79.52/18.08/2.40 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_min_vsby_3h | 710 | 0.102 | 79.44/17.89/2.68 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_min_ceiling_3h_ft | 699 | 0.1 | 79.54/18.03/2.43 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_precip_3h_in | 2,773 | 0.396 | 79.19/19.26/1.55 | 77.10/21.43/1.47 | 1.47 |  |
| test | dest_n_special_obs_3h | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | dest_alti_change_3h | 12,963 | 1.853 | 72.04/25.24/2.72 | 77.21/21.35/1.45 | 1.47 |  |
| test | dest_obs_age_min | 603 | 0.086 | 80.27/17.58/2.16 | 77.11/21.42/1.47 | 1.47 |  |
| test | tail_leg_index_hist | 327,816 | 46.872 | 76.19/22.42/1.39 | 77.93/20.53/1.54 | 1.47 |  |
| test | tail_sched_turn_min_hist | 399,766 | 57.16 | 78.19/20.44/1.37 | 75.68/22.72/1.60 | 1.47 |  |
| test | tail_prev_origin_hist | 399,766 | 57.16 | 78.19/20.44/1.37 | 75.68/22.72/1.60 | 1.47 |  |
| test | tail_prev_sched_elapsed_hist | 399,766 | 57.16 | 78.19/20.44/1.37 | 75.68/22.72/1.60 | 1.47 |  |

## 16. 서비스 재현 검증 (`build_service_features`)

학습셋에서 무작위 100편을 골라 `src/service_features.py`의 `build_service_features(flight, as_of_date=운항일)` 결과를 학습셋 값과 비교했습니다. 비교 피처 41개(날씨 제외).

- 모든 피처가 일치한 편: **100 / 100**
- 불일치 없음

## 17. v2 대비 변경점

| 항목 | v2 | v3 |
|---|---|---|
| BTS 공개 지연 규칙 | 없음 (직전 달·직전 주 BTS 사용) | 운항월 M → M−3월 말까지 (BTS_LAG_MONTHS) |
| hist_* 창 | M−12 ~ M−1 | M−14 ~ M−3 (source_max_month 기록) |
| 혼잡도 학습 피처 | *_proxy (D−7/14/21/28) | *_proxy_recent + origin_yoy_ratio (대안: alt_congestion_candidates) |
| v2 혼잡도 | 학습 피처 | *_proxy_v2_unavailable (병합본 전용) |
| 기체 일정 | 당일 Tail_Number 기준 tail_* 4개 (누수) | tail_*_hist 5개 (참조일 기준), 당일 값은 *_sameday (병합본 전용) |
| 결측 누수 검사 | 없음 | 모든 학습 피처 × split |
| 서비스 재현 | 없음 | build_service_features + 100편 대조 |
| aggregates | hist·sched_counts | + source_max_month/source_month, tail_sameday/, route_distance |
| 샘플·기상 | — | v2와 같은 행·같은 기상 값 (flight_id로 재사용) |
| 학습셋 행 수 | 6,559,804 | 6,559,804 |

## 18. 모델 담당에게 넘길 메모

- train은 정상편만 `NEG_RATE`=0.2로 줄였습니다. 그래서 모델이 낸 확률은 정상 확률이 실제보다 낮게 나옵니다.
- 되돌리는 방법은 둘 중 하나입니다.
  1. 학습할 때 `sample_weight`를 가중치로 쓰기 (`xgboost`: `DMatrix(..., weight=sample_weight)`)
  2. 예측 후 정상 클래스 확률에 `1/NEG_RATE`(=5)를 곱한 뒤, 세 클래스 합이 1이 되도록 다시 나누기
- 평가와 임계값 조정은 반드시 원래 분포인 validation/test로 하세요. 두 split은 클래스 비율을 건드리지 않고 균등하게만 줄였습니다.
- `sample_rate`, `sample_weight`는 학습 피처가 아닙니다. `feature_lists.json`의 목록만 피처로 쓰세요.
- 실험 순서 제안: `base` → `base_plus_candidates` → `with_dest_weather`. 일주일 전 예측은 `no_weather`. `experimental_tail`은 `tail_*_hist`(참조일 기준)라 v2의 당일 누수는 없지만, 효과는 별도 비교로 확인하세요.
- 범주형: `Reporting_Airline`, `Origin`, `Dest`, `route`, `tail_prev_origin_hist`.
- 학습셋의 BTS 기반 피처는 이미 공개 지연(M−3)을 반영했습니다. 서비스에서는 `build_service_features(flight, as_of_date)`로 `aggregates/`만 조회하면 같은 피처가 나옵니다. 새 BTS 월이 공개되면 `preprocess_flights.py`로 aggregates만 갱신하면 됩니다.
