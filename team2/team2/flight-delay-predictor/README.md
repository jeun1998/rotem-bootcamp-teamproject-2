# Flight Delay Predictor

BTS 항공 운항 데이터로 미국 국내선의 **지연**(도착 15분 이상)과 **결항** 확률을 예측하는 서비스.
용어 정의는 [CONTEXT.md](CONTEXT.md), 주요 결정은 [docs/adr/](docs/adr/) 참고.

## 구조

```
app/            Flask 서버 + 프론트 (static/index.html, app.js)
  predictor.py  예측기 인터페이스. 모델 파일이 없으면 더미 예측기 사용
  data/         드롭다운용 항공사·공항 목록 (임시값)
ml/             데이터 처리·학습 코드 (팀 작업 영역)
models/         학습된 모델 파일 (delay.joblib, cancel.joblib)
tests/          API 테스트
```

## 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
flask --app app run --debug     # http://localhost:5000
pytest
```

## API

- `GET /api/options` → `{ airlines: [{code, name}], airports: [{code, name}] }`
- `POST /api/predict` 요청 본문 `{ airline, origin, dest, date: "YYYY-MM-DD", dep_time: "HH:MM" }`
  - 200 → `{ delay_probability, cancel_probability, model_version }`
  - 400 → `{ errors: { <필드>: <메시지> } }`

## 모델 붙이기

`models/delay.joblib`과 `models/cancel.joblib`을 두고 서버를 재시작하면 더미 대신 실제 모델이 쓰입니다.
모델 파일이 지켜야 할 형식은 [ADR-0001](docs/adr/0001-model-artifact-includes-preprocessing.md)에 있습니다.
