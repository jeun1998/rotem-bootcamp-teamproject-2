"""운영체제와 상관없이 열리는 XGBoost 모델 묶음.

XGBoost 를 그대로 피클하면 난수 생성기 상태(rng_state)가 C++ 표준 라이브러리 형식의 문자열로 함께 저장된다.
Linux 에서 저장한 형식을 Windows 의 XGBoost 가 읽지 못해 'input stream corrupted' 로 실패했다.
이 클래스는 피클에 XGBoost 모델을 공식 UBJSON 모델 형식(실행 상태 없음)으로 담고, 불러올 때 다시 만든다.
"""
from __future__ import annotations


class PortableXGBPipeline:
    """sklearn Pipeline(전처리 + 마지막 단계 XGBClassifier)을 감싼다. predict_proba, classes_ 는 그대로 쓴다."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @property
    def classes_(self):
        return self.pipeline.classes_

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def __getstate__(self):
        *head, (name, clf) = self.pipeline.steps
        return {"head": head, "name": name, "params": clf.get_params(),
                "booster_ubj": bytes(clf.get_booster().save_raw("ubj"))}

    def __setstate__(self, state):
        from sklearn.pipeline import Pipeline
        from xgboost import XGBClassifier

        clf = XGBClassifier(**state["params"])
        clf.load_model(bytearray(state["booster_ubj"]))
        self.pipeline = Pipeline(state["head"] + [(state["name"], clf)])
