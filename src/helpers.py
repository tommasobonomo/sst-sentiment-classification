from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import wandb
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from settings import ClassifierType, Config, PreprocessorType, console_logger
from src.classifiers import TransformerPredictor, XGBoostPredictor
from src.classifiers.deep_models import TransformerModule
from src.preprocessors import TfIdfPreprocessor, TransformerTokenizer


def init_from_hydra_config(config: Config, read_data: bool = True):
    if read_data:
        console_logger.info("Read dataset...")

        dataset = pd.read_csv(config.dataset_path)

        console_logger.info("Prepare splits...")

        train_set, dev_set, test_set = (
            dataset[dataset["split"] == "train"],
            dataset[dataset["split"] == "dev"],
            dataset[dataset["split"] == "test"]
        )

        if config.merge_dev_with_train:
            train_set = pd.concat([train_set, dev_set], axis=0)

    console_logger.info("Configuring training pipeline...")

    if config.preprocessor == PreprocessorType.tfidf:
        preprocessor = TfIdfPreprocessor(config.tfidf_config)
    elif config.preprocessor == PreprocessorType.transformer_tokenizer:
        preprocessor = TransformerTokenizer(config.transformer_name, config.transformer_tokenizer_config)
    else:
        raise ValueError(f"Unsupported preprocessor {config.preprocessor}")

    if config.classifier == ClassifierType.xgboost:
        classifier = XGBoostPredictor(config.xgboost_config)
    elif config.classifier == ClassifierType.transformer:
        classifier = TransformerPredictor(config.transformer_name, config.transformer_predictor_config)
    else:
        raise ValueError(f"Unsupported classifier {config.classifier}")

    if read_data:
        return train_set, dev_set, test_set, preprocessor, classifier
    else:
        return preprocessor, classifier


@lru_cache
def get_inference_pipeline(classifier_type: ClassifierType) -> Pipeline:
    # Disable W&B
    wandb.init(mode="disabled")

    model_path = Path("model")
    raw_config: dict = OmegaConf.load(model_path / "config.yaml")  # type: ignore
    config = Config(**raw_config)

    if classifier_type == ClassifierType.xgboost:
        inference_pipeline = joblib.load(model_path / "xgboost" / "pipeline.joblib")
    elif classifier_type == ClassifierType.transformer:
        preprocessor, classifier = init_from_hydra_config(config, read_data=False)
        classifier.module = TransformerModule.load_from_checkpoint(model_path / "transformer" / "model.ckpt")
        inference_pipeline = Pipeline([
            ("preprocessor", preprocessor), ("classifier", classifier)
        ])
    else:
        raise ValueError(f"Unsupported classifier {classifier_type}")

    return inference_pipeline
