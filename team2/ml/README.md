# ml/

모델 담당이 보는 안내. 데이터 처리와 학습 데이터 생성은 데이터 팀 코드와 문서가 기준이다.

- `weather_download/`: IEM 공항 기상(METAR) 다운로드·검증
- `data_pipeline/`: BTS + 기상 결합, 피처 생성, 학습셋(`processed_v3`) 생성 → [README](data_pipeline/README.md)
- `data_pipeline/meta/`: `feature_lists.json`, 품질·피처 선택 리포트

## 데이터

- 학습셋은 데이터 폴더(`AIRLINE_DATA_ROOT`)의 `processed_v3/ml/split=train|validation|test/part-YYYY-MM.parquet`. 용량 때문에 저장소 밖에서 따로 받는다(약 377 MB). 피처 목록은 저장소의 `data_pipeline/meta/feature_lists.json`이다.
- 기간과 분할: train 2022~2023년, validation 2024년, test 2025년 운항.
- 레이블 `label`: 0 정상, 1 지연(출발 15분 이상), 2 결항. 용어는 [CONTEXT.md](../CONTEXT.md).
- train은 정상편만 20%로 줄였다(정상 40.8% / 지연 54.0% / 결항 5.2%). validation과 test는 원본 비율 그대로다.

## 학습 규칙

- 피처는 `feature_lists.json`의 목록에서만 고른다. 지금 서비스는 기상 없는 `no_weather` 목록을 쓴다. 기상 피처는 서비스에서 만들 수 없어서, 들어 있으면 서버가 시작하지 않는다.
- 학습 금지 컬럼(실제 시각·지연, 결항 사유, `label_arr`, `sample_rate`, `sample_weight`, `*_oracle`, `*_sameday` 등)은 쓰지 않는다.
- 결측 대체, 인코딩, 스케일링은 train에만 fit한다. 결측은 NaN 그대로 둬도 된다(XGBoost가 처리).
- 확률 보정: `sample_weight`로 학습하고 `normal_sample_rate: 1.0`으로 저장한다(데이터 팀 기본). 가중치 없이 학습했다면 `0.2`로 저장하면 서버가 보정한다.
- 평가는 validation·test에서 클래스별 PR-AUC·Recall, macro F1, 로그 손실로 본다. 결항이 1~2%라 정확도는 의미가 없다.
- 실험 순서는 `base` → `base_plus_candidates` → `with_dest_weather` → `experimental_tail`을 권한다. 서비스에 붙일 모델은 그중 `no_weather`로 다시 학습한다.

## 모델 파일

지금은 `team2/scripts/train.py`가 저장한 `outputs/models/no_weather.joblib`과 `outputs/metadata.json`을 `ml/package_model.py`로 묶어 `models/model.joblib`을 만든다. 아래처럼 학습 코드에서 바로 저장하면 이 단계가 없어진다.

산출물은 `models/model.joblib` 하나다. 계약은 [ADR-0002](../docs/adr/0002-service-features-and-model-artifact.md)를 따른다.

```python
lists = json.load(open("ml/data_pipeline/meta/feature_lists.json", encoding="utf-8"))
feats = lists["no_weather"]                      # 지금 서비스는 기상 없는 모델
cats = [c for c in lists["categorical"] if c in feats]
for c in cats:
    train[c] = train[c].astype("category")
model = xgb.XGBClassifier(enable_categorical=True, ...).fit(train[feats], train["label"], sample_weight=train["sample_weight"])
joblib.dump({
    "model": model,                              # predict_proba 열 순서 = [정상, 지연, 결항]
    "features": feats,
    "categories": {c: list(train[c].cat.categories) for c in cats},
    "normal_sample_rate": 1.0,                   # sample_weight 없이 학습했다면 0.2
    "temperature": t,                            # 검증셋 온도 보정값 (선택, 없으면 1.0)
    "version": "xgb-no_weather-2026-11-18",
}, "models/model.joblib")
```

## 팀 합의가 필요한 항목 (미정)

- 기상 포함 모델(출발 3시간 전 예측)을 서비스에 붙일지와, 실시간 METAR를 어디서 받을지
- 예측 가능 기간(지금 365일) 안의 먼 미래 항공편 성능 (학습셋은 운항월과 BTS 기준월 간격이 항상 3개월)
