from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


@dataclass
class TfIdfConfig:
    output_dims: int = 50  # Final dimension of the output vector that represents a sample


class TfIdfPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: TfIdfConfig = TfIdfConfig()) -> None:
        super().__init__()
        self.config = config

        self.pipeline = Pipeline([
            ("tf-idf", TfidfVectorizer(strip_accents="unicode")),
            ("dimensionality_reduction", TruncatedSVD(n_components=config.output_dims))
        ])

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.pipeline.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self.pipeline)

        return self.pipeline.transform(X)
