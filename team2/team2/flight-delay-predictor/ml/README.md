# ml/

데이터 수집 → 전처리 → 학습 코드를 두는 곳.

산출물은 `models/delay.joblib`, `models/cancel.joblib` 두 개이며,
**전처리를 포함한 sklearn Pipeline**이어야 합니다. 입력 컬럼 계약은
[ADR-0001](../docs/adr/0001-model-artifact-includes-preprocessing.md)을 따릅니다.

```python
pipe = Pipeline([("prep", ColumnTransformer([...])), ("clf", ...)])
pipe.fit(df[FEATURE_COLUMNS], df["ArrDel15"])
joblib.dump(pipe, "models/delay.joblib")
```

`FEATURE_COLUMNS`는 `app/predictor.py`에서 import해서 쓰면 서비스와 어긋나지 않습니다.

## 팀 합의가 필요한 항목 (미정)

- 예측 시점: 예약 시점 vs 출발 당일 (출발 후에야 알 수 있는 컬럼은 feature 금지)
- 타깃 정의: 지연 기준(ArrDel15 vs DepDel15), 모델 2개 vs 3-class, 회항 처리
- 데이터 범위: 기간(코로나 시기 포함 여부), 공항·항공사 범위 → 범위가 정해지면 `app/data/*.json` 갱신
