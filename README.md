# delay-cast

BTS 항공 운항 데이터로 미국 국내선의 **지연**(출발 15분 이상)과 **결항** 확률을 예측하는 서비스.
용어 정의는 [CONTEXT.md](CONTEXT.md), 주요 결정은 [docs/adr/](docs/adr/) 참고.

## 구조

```
app/            Flask 서버 + 프론트 (static/index.html, app.js, style.css, video/)
  static/vendor/  달력 라이브러리 Air Datepicker 3.6.0 (MIT, npm 배포본 그대로)
  predictor.py  예측기. models/model.joblib 이 없으면 더미 예측기 사용
  data/         드롭다운 목록 airlines.json·airports.json·routes.json (ml/build_options.py 가 생성), 이름표 names.json
ml/             모델 담당 안내
  data_pipeline/  데이터 팀 파이프라인. service_features.py 가 서비스 피처를 만든다
  weather_download/  공항 기상 다운로드, station_map.csv (공항 시간대)
data/aggregates/  서비스 피처용 BTS 집계표 (데이터 팀이 갱신)
models/         학습된 모델 파일 (model.joblib, 커밋하지 않음)
tests/          API 테스트, 모델 계약(ADR-0002) 테스트
docs/adr/       주요 결정 기록
docs/design/    화면 스크린샷
```

서비스 피처에 필요한 코드와 데이터(`ml/data_pipeline/`, `data/aggregates/`, `station_map.csv`)는 저장소에 있습니다. 실제 모델로 서버를 띄우려면 `models/model.joblib`([ADR-0002](docs/adr/0002-service-features-and-model-artifact.md) 형식)만 받으면 됩니다. 학습셋과 BTS 원본은 용량 때문에 저장소 밖에 있습니다([ml/data_pipeline/README.md](ml/data_pipeline/README.md)).

집계표는 새 BTS 달이 공개될 때마다 데이터 팀이 `preprocess_flights.py`로 갱신합니다. 갱신이 밀리면 예측 가능한 출발일이 줄어들고, 달력 아래쪽에 그 이유가 보입니다.

드롭다운의 항공사·공항은 집계표에서 최근 12개월 운항 기록이 있는 곳을 운항이 많은 순으로 담습니다. 세 칸은 실제 운항 노선(`routes.json`)으로 서로 걸러집니다. 예를 들어 항공사를 고르면 출발지에는 그 항공사가 출발하는 공항만, 도착지에는 그 노선만 보이고, 서버도 운항하지 않는 조합은 받지 않습니다. 집계표에 항공사×노선 표가 없어 노선은 기체 일정(`tail_sameday`)에서 만듭니다(1년 12회 이상 운항한 구간, 2025년 운항의 99.98%). 집계표를 갱신한 뒤 아래를 실행해 목록도 갱신합니다. 한글 이름은 `app/data/names.json`(항공사 전부, 상위 공항 100곳)에서 붙이고, 나머지 공항은 영문 이름만 씁니다. 새 항공사가 생기면 스크립트가 이름표에 추가하라고 알려 줍니다.

```bash
python ml/build_options.py
```

상단 배경 영상(`app/static/video/sky-loop.mp4`)은 Coverr의 "Airplane flying through the clouds"를 반복 재생용으로 이어 붙이고 1280px로 압축한 것입니다. Coverr 라이선스로 출처 표기 없이 사용할 수 있습니다.

## 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
flask --app app run --debug     # http://localhost:5000
pytest
```

## API

- `GET /api/options` → `{ airlines: [{code, name, name_ko?}], airports: [{code, name, name_ko?}], routes: {항공사: {출발: [도착]}}, dates: {min, max, limit_note} }`
  - `dates.max`는 예측 가능한 마지막 출발일. 집계표가 오래돼 365일보다 짧아지면 `limit_note`에 이유가 담긴다
- `POST /api/predict/hourly` 같은 요청 본문 → `{ model_version, hours: [{hour, delay_probability, cancel_probability}] }` (06~22시, 예정 소요시간 유지)
- `POST /api/predict` 요청 본문 `{ airline, origin, dest, date: "YYYY-MM-DD", dep_time: "HH:MM", arr_time: "HH:MM" }` (시각은 각 공항 현지 시각)
  - 200 → `{ delay_probability, cancel_probability, model_version }`
  - 400 → `{ errors: { <필드>: <메시지> } }` (예측 가능 기간 밖의 날짜도 `date` 오류)
  - 500 → `{ errors: { _: <메시지> } }` (피처 생성이나 모델 예측 실패)

## 모델 붙이기

서비스 모델 `models/model.joblib`은 저장소에 커밋되어 있어, 받으면 바로 실제 모델로 예측합니다(이 파일이 없으면 더미 예측기). 모델 팀이 새로 학습하면 `no_weather.joblib`과 `metadata.json` 두 파일을 `models/`에 넣고 아래를 실행한 뒤, 바뀐 `models/model.joblib`만 커밋합니다. 두 원본 파일은 커밋되지 않습니다.

```bash
python ml/package_model.py models     # → models/model.joblib
```

두 파일은 `machine` 브랜치 `team2/outputs/`에 Git LFS로 있습니다. 브랜치를 꺼낼 때는 `GIT_LFS_SKIP_SMUDGE=1`로 LFS 자동 받기를 끄고 두 파일만 `git lfs pull --include`로 받으세요. 그대로 꺼내면 쓰지 않는 444MB 모델까지 받습니다.
 피처는 `ml/data_pipeline/service_features.py`(데이터 팀 `data-mgb` 브랜치)로 만들고, 다른 경로를 쓰려면 `FLIGHT_FEATURES_SRC`에 지정합니다.
서버는 그 폴더의 `service_features.build_service_features`로 피처를 만들고, `aggregates/`와 `station_map.csv`는 데이터 팀 `config`가 가리키는 경로에서 읽습니다.
모델 파일이 계약에 맞지 않거나 `service_features.py`를 찾지 못하면 서버가 시작하지 않고 이유를 알려 줍니다.

```bash
flask --app app run                                        # ml/data_pipeline 사용
FLIGHT_FEATURES_SRC=/other/path/to/data_pipeline flask --app app run   # 다른 경로를 쓸 때
```

모델 파일 형식과 이유는 [ADR-0002](docs/adr/0002-service-features-and-model-artifact.md)에 있습니다.
