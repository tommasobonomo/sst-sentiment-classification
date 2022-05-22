from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier


@dataclass
class XGBoostConfig:
    n_estimators: int = 5


class XGBoostPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

        self.model = XGBClassifier(**config)

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.round(y)

        self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
