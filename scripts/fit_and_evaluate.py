import hydra
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from settings import ClassifierType, Config, PreprocessorType, console_logger
from src.classifiers import TransformerPredictor, XGBoostPredictor
from src.preprocessors import TfIdfPreprocessor, TransformerTokenizer

config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def run(config: Config) -> None:
    console_logger.info("Read dataset...")

    dataset = pd.read_csv(config.dataset_path)

    console_logger.info("Prepare splits...")

    train, dev, test = dataset[dataset["split"] == "train"], dataset[dataset["split"] == "dev"], dataset[dataset["split"] == "test"]

    if config.merge_dev_with_train:
        train = pd.concat([train, dev], axis=0)

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

    train_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    console_logger.info("Running training...")

    y_train = train["label"].values.round()
    X_train = train["sentence"].values
    train_pipeline.fit(X_train, y_train)

    if config.evaluate_on_dev:
        console_logger.info("Running prediction on dev set...")
        test_set = dev
    else:
        console_logger.info("Running prediction on test set...")
        test_set = test

    y_test = test_set["label"].values
    X_test = test_set["sentence"].values
    y_pred = train_pipeline.predict(X_test)

    report = classification_report(y_test.round(), y_pred, digits=4)
    print(report)


if __name__ == "__main__":
    run()
