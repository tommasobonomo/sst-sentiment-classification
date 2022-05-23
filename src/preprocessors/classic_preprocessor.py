from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


@dataclass
class TfIdfConfig:
    output_dims: int = 49           # Final dimension of the output vector that represents a sample
    # If TruncatedSVD should be used on the output of Tf-idf.
    # If it is, the process is also called Latent Semantic Indexing (https://en.wikipedia.org/wiki/Latent_semantic_analysis#Latent_semantic_indexing)
    use_truncated_svd: bool = True
    min_ngram_range: int = 1
    max_ngram_range: int = 2


class TfIdfPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: TfIdfConfig = TfIdfConfig()) -> None:
        super().__init__()
        self.config = config

        ngram_range = (config.min_ngram_range, config.max_ngram_range)

        if config.use_truncated_svd:
            self.pipeline = Pipeline([
                ("tf-idf", TfidfVectorizer(ngram_range=ngram_range, strip_accents="unicode")),
                ("dimensionality_reduction", TruncatedSVD(n_components=config.output_dims))
            ])
        else:
            self.pipeline = Pipeline([
                ("tf-idf", TfidfVectorizer(ngram_range=ngram_range, strip_accents="unicode")),
            ])

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.pipeline.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.transform(X)
