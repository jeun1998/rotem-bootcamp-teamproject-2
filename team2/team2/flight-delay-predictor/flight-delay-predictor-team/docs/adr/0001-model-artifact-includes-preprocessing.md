# 모델 파일은 전처리를 포함한 sklearn Pipeline 통째로 저장한다

학습 쪽(`ml/`)과 서비스 쪽(`app/`)은 서로 다른 사람이 동시에 개발한다. 전처리(원-핫 인코딩, 시간대 구간화 등)를 Flask에서 따로 구현하면 학습 코드와 이중 구현이 되고, 둘이 어긋나도 에러 없이 틀린 확률이 나온다. 그래서 모델 파일은 **BTS 원본 컬럼 이름 그대로의 DataFrame**을 입력으로 받는 sklearn `Pipeline`(전처리 + 분류기)으로 `joblib` 저장하기로 했다. Flask는 컬럼만 맞춰서 `predict_proba`를 호출하고, 전처리에 대해서는 아무것도 모른다.

## 계약

- `models/delay.joblib`: 지연(`ArrDel15`) 이진 분류 Pipeline
- `models/cancel.joblib`: 결항(`Cancelled`) 이진 분류 Pipeline
- 둘 다 아래 컬럼을 가진 DataFrame을 받고, `predict_proba(X)[:, 1]`이 양성 확률이어야 한다.

| 컬럼 | 타입 | 예시 |
|---|---|---|
| `FlightDate` | str (YYYY-MM-DD) | `2024-07-04` |
| `Month` | int 1–12 | `7` |
| `DayofMonth` | int 1–31 | `4` |
| `DayOfWeek` | int 1(월)–7(일) | `4` |
| `CRSDepTime` | int hhmm | `1730` |
| `Reporting_Airline` | str | `AA` |
| `Origin` | str (IATA) | `JFK` |
| `Dest` | str (IATA) | `LAX` |

Pipeline이 이 중 일부만 써도 괜찮다. 다른 컬럼(예: `Distance`)이 필요하면 이 표를 먼저 고쳐야 한다. 그 컬럼은 `app/predictor.py`의 `FEATURE_COLUMNS`와 함께 고친다.

## 고려한 대안

- **Flask에서 전처리 후 숫자 벡터만 모델에 전달**: 이중 구현이 생기고 어긋날 위험이 있어 기각했다.
- **ONNX 등 프레임워크 중립 포맷**: 부트캠프 범위에서는 과하다고 판단해 기각했다.
