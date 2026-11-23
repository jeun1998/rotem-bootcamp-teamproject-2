# 서버가 데이터 팀의 서비스 피처 함수로 피처를 만들고, 모델 파일은 메타데이터와 함께 저장한다

ADR-0001은 모델이 BTS 원본 8개 컬럼을 받고 전처리를 통째로 품는다고 정했다. 데이터셋 v3의 피처는 과거 실적(`hist_*`), 혼잡도(`*_proxy_recent`), 작년 대비 추세처럼 BTS 집계표(`aggregates/`)를 조회해야 만들 수 있어서 모델 파일 안에 넣을 수 없다. 데이터 팀은 학습셋과 같은 값을 만드는 `build_service_features`를 제공하고, 무작위 100편에서 학습셋과 100% 일치함을 검증했다. 그래서 서버는 이 함수로 피처를 만들고 모델에는 피처만 넣는다.

## 계약

**피처 생성.** 서버는 `ml/data_pipeline/`(다른 경로는 `FLIGHT_FEATURES_SRC`)의 데이터 팀 `service_features`를 import한다.

- `build_service_features(flight, as_of_date)`에 BTS 컬럼 이름으로 한 편을 넘긴다. `CRSDepTime`, `CRSArrTime`은 현지 시각 4자리 hhmm(`"0705"`), `as_of_date`는 요청을 받은 날이다.
- 편명(`Flight_Number_Reporting_Airline`)은 빈 문자열로 넘긴다. 편명은 기체 일정 피처(`tail_*_hist`)에만 쓰이고 지금 모델은 그 피처를 쓰지 않는다.
- 함수가 돌려주지 않은 피처는 결측(NaN)으로 둔다. 학습셋에서도 참조 데이터가 없으면 결측이다.
- `features=`에 모델의 피처 목록을 넘겨, 모델이 쓰지 않는 단계(기체 일정 등)는 계산하지 않게 한다.
- 서버는 `build_service_features_batch`로 피처를 만든다. 시간대별 비교의 17편은 한 번에 넘기고 모델도 한 번에 예측한다(한 편씩 17번이면 2.9초, 한 번에 0.2초 안팎).
- 함수가 `StaleAggregatesError`를 내면(집계표에 필요한 달이 없음) 서버는 그 날짜를 예측할 수 없다고 400으로 답한다. 예측 가능한 마지막 날은 달마다 한 편씩 함수를 불러 처음 실패하는 달로 알아낸다. 기준월 규칙을 서버에 옮겨 적지 않으므로, 데이터 팀 규칙이나 집계표가 바뀌어도 그대로 맞는다.

**모델 파일.** `models/model.joblib`은 다음 키를 가진 dict다. 서버는 시작할 때 검증하고, 어기면 시작하지 않는다.

| 키 | 내용 |
|---|---|
| `model` | `predict_proba(X)`의 열이 `[정상, 지연, 결항]`(레이블 0, 1, 2) 순서인 분류기 |
| `features` | 모델 입력 컬럼 목록. 모두 `service_feature_names()` 안에 있어야 한다 |
| `categories` | 범주형 컬럼별 학습 때의 범주 목록 |
| `normal_sample_rate` | 학습에 쓴 정상편 샘플링 비율. `sample_weight`로 학습했으면 1.0 |
| `version` | 화면과 API에 나가는 모델 이름 |
| `temperature` (선택) | 검증셋에서 맞춘 온도 보정값. 서버는 `softmax(log p / T)`를 먼저 적용한다. 없으면 1.0 |

모델 팀 학습 결과(`no_weather.joblib` Pipeline + `metadata.json`)는 `ml/package_model.py`로 이 형식에 묶는다. 범주형은 모델 Pipeline의 인코더가 처리하므로 `categories`는 비워 둔다. XGBoost 모델은 `app/portable_model.py`로 감싸 XGBoost 공식 UBJSON 모델 형식으로 담는다. 그대로 피클하면 운영체제마다 형식이 다른 난수 상태(`rng_state`)가 함께 저장되어, Linux에서 묶은 모델이 Windows에서 'input stream corrupted'로 열리지 않았다. 모델 팀이 `train.py`에서 이 형식으로 바로 저장하게 되면 스크립트는 지운다.

**범주형 처리.** 서버는 범주형 컬럼을 `categories`의 목록으로 `pandas.Categorical`을 만들어 넘기고, 목록에 없는 값은 결측으로 둔다. 한 행짜리 DataFrame을 그냥 `category`로 바꾸면 어떤 값이든 코드 0이 된다. XGBoost 2.1.4에서 재현했을 때 JFK와 ORD가 모두 ATL로 취급되어 에러 없이 틀린 확률이 나왔다. 3.4.1은 범주를 모델에 저장해 맞춰 주지만, 이 방식은 버전과 상관없이 맞다.

**확률 보정.** 학습셋은 정상편을 `normal_sample_rate`만 뽑았다. 서버는 정상 확률을 그 비율로 나눈 뒤 세 확률을 다시 정규화하고, 지연·결항 확률을 보여준다.

## 고려한 대안

- **ADR-0001 유지(원본 컬럼 Pipeline)**: 집계표 조회를 Pipeline에 넣으면 모델 파일이 데이터 경로에 묶이고 크기가 커진다. 기각했다.
- **서버가 피처를 직접 구현**: 학습 코드와 이중 구현이 되어 어긋날 위험이 있다. 데이터 팀 함수를 그대로 쓴다.
- **확률 보정을 모델 쪽에만 맡김**: 잊어도 에러가 나지 않아 확률이 조용히 부풀려진다. 비율을 필수 키로 받아 서버가 적용한다.
