from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class XGBoostConfig:
    type: Literal["regressor", "classifier"] = "regressor"
    n_estimators: int = 5


class XGBoostPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, config: XGBoostConfig) -> None:
        super().__init__()

        self.config = config

        dict_config = asdict(config)

        if dict_config.pop("type") == "regressor":
            self.model = XGBRegressor(**dict_config)
        else:
            self.model = XGBClassifier(**dict_config)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.config.type == "classifier":
            y = np.round(y)

        self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.config.type == "regressor":
            return self.model.predict(X)
        else:
            # self.config.type == "classifier"
            return self.model.predict_proba(X)
