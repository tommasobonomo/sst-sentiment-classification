from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from settings import ClassifierType, Config, console_logger
from src.helpers import init_from_hydra_config

config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def run(config: Config) -> None:
    train, dev, test, preprocessor, classifier = init_from_hydra_config(config)

    if config.merge_dev_with_train:
        train = pd.concat([train, dev], axis=0)

    train_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    console_logger.info("Running training...")

    y_train = train["label"].values.round()
    X_train = train["sentence"].values
    train_pipeline.fit(X_train, y_train)

    if config.save_model:
        save_path = Path("model")
        console_logger.info(f"Saving model to directory {save_path.as_posix()}...")
        save_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, save_path / "config.yaml")

        if config.classifier == ClassifierType.xgboost:
            xgboost_path = save_path / "xgboost"
            xgboost_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(train_pipeline, xgboost_path / "pipeline.joblib")
        elif config.classifier == ClassifierType.transformer:
            transformer_path = save_path / "transformer"
            best_model_path = Path(classifier.trainer.checkpoint_callback.best_model_path)
            best_model_path.rename(transformer_path / "model.ckpt")
        else:
            raise ValueError(f"Unsupported classifier {config.classifier}")

    if config.evaluate_on_dev:
        console_logger.info("Running prediction on dev set...")
        test_set = dev
    else:
        console_logger.info("Running prediction on test set...")
        test_set = test

    y_test = test_set["label"].values
    X_test = test_set["sentence"].values
    y_pred_proba = train_pipeline.predict(X_test)

    y_pred = np.argmax(y_pred_proba, axis=1)

    report = classification_report(y_test.round(), y_pred, digits=4)
    print(report)
    score = f1_score(y_test.round(), y_pred)
    return 1 - score


if __name__ == "__main__":
    run()
