import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.classifiers import TransformerPredictorConfig, XGBoostConfig
from src.preprocessors import TfIdfConfig, TransformerTokenizerConfig

console_logger = logging.getLogger(__name__)


class PreprocessorType(str, Enum):
    tfidf = "tfidf"
    transformer_tokenizer = "transformer_tokenizer"


class ClassifierType(str, Enum):
    xgboost = "xgboost"
    transformer = "transformer"


@dataclass
class Config:
    preprocessor: PreprocessorType = PreprocessorType.transformer_tokenizer
    classifier: ClassifierType = ClassifierType.transformer
    dataset_path: Path = Path("data") / "labelled_sentences.csv"
    merge_dev_with_train: bool = True
    evaluate_on_dev: bool = False
    transformer_name: str = "distilbert-base-uncased"
    tfidf_config: TfIdfConfig = TfIdfConfig()
    transformer_tokenizer_config: TransformerTokenizerConfig = TransformerTokenizerConfig()
    xgboost_config: XGBoostConfig = XGBoostConfig()
    transformer_predictor_config: TransformerPredictorConfig = TransformerPredictorConfig()
