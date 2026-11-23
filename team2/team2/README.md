# 항공편 지연·결항 예측 — v3

BTS + IEM 데이터 / XGBoost · Random Forest · Logistic Regression

## 데이터
train: 2022~2023 / validation: 2024 / test: 2025.
ML 데이터는 processed_v3/ml/ 의 월별 Parquet만 사용합니다.
merged와 interim 파일은 분석용이며 직접 학습에 넣지 않습니다.

## 준비
1. 압축을 풀어 team2_flight_v3 폴더를 만듭니다.
2. 기존에 업로드한 ml.zip, split=test.zip, split=validation.zip을
   data/inbox/ 폴더에 복사합니다.
3. 필요하면 feature_lists(1).json도 inbox에 넣습니다.

## 설치 및 실행
conda activate myproject
cd C:\works\projects\team2_flight_v3
python -m pip install -r requirements.txt
python scripts/import_uploads.py
python scripts/check_data.py
python scripts/train.py --max-train-rows 200000
python scripts/evaluate.py
python app/server.py

브라우저 주소: http://127.0.0.1:5000

## 주의
sample_weight는 모델 학습용 가중치일 뿐 예측 입력이 아닙니다.
정상편 사후 확률 5배 재보정은 하지 않습니다.
최종 모델 선정은 2024년 validation의 결항 PR-AUC 기준이며,
2025년 test는 최종 평가에 사용합니다.
코드는 작성본이며 실제 대용량 전체 학습 검증은 아직 진행하지 않았습니다.
