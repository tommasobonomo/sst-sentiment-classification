from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer


@dataclass
class TransformerTokenizerConfig:
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    return_attention_mask: bool = True


class TransformerTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_model: str, config: TransformerTokenizerConfig = TransformerTokenizerConfig()) -> None:
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        return self

    def transform(self, X: np.ndarray) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            X.tolist(),
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=self.config.return_tensors,
            return_attention_mask=self.config.return_attention_mask
        )
