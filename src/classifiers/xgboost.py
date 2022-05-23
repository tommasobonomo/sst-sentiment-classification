from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier


@dataclass
class XGBoostConfig:
    n_estimators: int = 95
    max_depth: int = 7


class XGBoostPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, config: XGBoostConfig) -> None:
        super().__init__()

        self.config = config

        self.model = XGBClassifier(**config)  # type: ignore

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
