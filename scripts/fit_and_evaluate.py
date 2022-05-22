from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from helpers.logging import console_logger
from src.classifiers import XGBoostConfig, XGBoostPredictor
from src.preprocessors import TfIdfConfig, TfIdfPreprocessor


class PreprocessorType(str, Enum):
    tfidf = "tfidf"
    bpe = "bpe"


class ClassifierType(str, Enum):
    xgboost = "xgboost"
    transformer = "transformer"


@dataclass
class Config:
    preprocessor: PreprocessorType = PreprocessorType.tfidf
    classifier: ClassifierType = ClassifierType.xgboost
    dataset_path: Path = Path("data") / "labelled_sentences.csv"
    tfidf_config: TfIdfConfig = TfIdfConfig()
    xgboost_config: XGBoostConfig = XGBoostConfig()
    merge_dev_with_train: bool = False
    evaluate_on_dev: bool = False


config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def run(config: Config) -> None:
    console_logger.info("Read dataset...")

    dataset = pd.read_csv(config.dataset_path)

    console_logger.info("Prepare splits...")

    train, dev, test = dataset[dataset["split"] == "train"], dataset[dataset["split"] == "dev"], dataset[dataset["split"] == "test"]

    console_logger.info("Configuring training pipeline...")

    if config.preprocessor == PreprocessorType.tfidf:
        preprocessor = TfIdfPreprocessor(config.tfidf_config)
    elif config.preprocessor == PreprocessorType.bpe:
        raise NotImplementedError("BPE preprocessor not implemented yet")
    else:
        raise ValueError(f"Unsupported preprocessor {config.preprocessor}")

    if config.classifier == ClassifierType.xgboost:
        classifier = XGBoostPredictor(config.xgboost_config)
    elif config.classifier == ClassifierType.transformer:
        raise NotImplementedError("Transformer classifier not implemented yet")
    else:
        raise ValueError(f"Unsupported classifier {config.classifier}")

    train_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    console_logger.info("Running training...")

    y_train = train["label"].values
    X_train = train["sentence"].values
    train_pipeline.fit(X_train, y_train)

    console_logger.info("Running prediction on test set...")
    y_test = test["label"].values
    X_test = test["sentence"].values
    y_pred = train_pipeline.predict(X_test)

    print(classification_report(y_test.round(), y_pred))


if __name__ == "__main__":
    run()
