from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

LabelledBatch = Tuple[BatchEncoding, torch.Tensor]
Batch = BatchEncoding


@dataclass
class TransformerPredictorConfig:
    learning_rate: float = 1e-5
    batch_size: int = 2
    val_fraction: float = 0.2
    fast_dev_run: bool = False
    epochs: int = 10
    pooling_strategy: str = "pooler_output"  # ["pooler_output", "mean_masked_pooling"]
    num_workers: int = 4


class TransformerPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, transformer_model: str, config: TransformerPredictorConfig = TransformerPredictorConfig()) -> None:
        super().__init__()
        self.config = config
        self.module = TransformerModule(
            transformer_model,
            learning_rate=config.learning_rate,
            pooling_strategy=config.pooling_strategy
        )

        self.callbacks = [EarlyStopping(monitor="val_loss")]
        self.loggers = [WandbLogger(project="sentiment-classifier")]

    def fit(self, X: Dict[str, torch.Tensor], y: np.ndarray):
        # We assume that X is as outputted by a Huggingface tokenizer, i.e. a dict with keys "input_ids" and "attention_mask"
        # y should be a rounded array containing either 0 or 1
        dataset = TrainingDataset(X, y)
        val_cardinality = round(len(dataset) * self.config.val_fraction)
        train_cardinality = len(dataset) - val_cardinality
        train_dataset, val_dataset = random_split(dataset, [train_cardinality, val_cardinality])

        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

        self.trainer = pl.Trainer(
            accelerator="auto",
            fast_dev_run=self.config.fast_dev_run,
            max_epochs=self.config.epochs
        )
        self.trainer.fit(self.module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        return self

    def predict(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        dataset = Dataset(X)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

        with torch.no_grad():
            raw_predictions = self.trainer.predict(self.module, dataloaders=dataloader, return_predictions=True)

        prediction_scores = torch.cat(raw_predictions)  # type: ignore

        # As we are simply predicting the best of two classes, we don't need to pass the predictions through a softmax
        # and can argmax directly on the logits
        return torch.argmax(prediction_scores, dim=1)


class TransformerModule(pl.LightningModule):
    def __init__(self, transformer_model: str, learning_rate: float, pooling_strategy: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_model)
        self.classifier_head = torch.nn.Linear(self.encoder.config.hidden_size, 2)

        self.save_hyperparameters()

    def _transformers_output_to_pooled(
        self,
        output: BaseModelOutputWithPoolingAndCrossAttentions,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.hparams.pooling_strategy == "pooler_output":  # type: ignore
            # Take embedding of first token, corresponding to `[CLS]` token, as pooled representation
            return output.last_hidden_state[:, 0, :]
        elif attention_mask is not None and self.hparams.pooling_strategy == "mean_masked_pooling":  # type: ignore
            # Average the embedding of all non-masked representations of sequence tokens
            last_layer = output.last_hidden_state
            attention_mask = attention_mask.unsqueeze(-1)
            non_masked_mean = torch.sum(attention_mask * last_layer, dim=1) / attention_mask.sum(dim=1)
            return non_masked_mean
        else:
            raise RuntimeError("`self.hparams.pooling_strategy` should be one of `pooler_output` and `mean_masked_pooling`")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def forward(self, batch: Batch) -> torch.Tensor:  # type: ignore
        raw_outs = self.encoder(**batch, return_dict=True)
        pooled_outs = self._transformers_output_to_pooled(raw_outs, batch.get("attention_mask"))
        return self.classifier_head(pooled_outs)

    def step(self, batch: LabelledBatch) -> torch.Tensor:
        sentences, labels = batch

        prediction_logits = self.forward(sentences)

        loss = F.cross_entropy(prediction_logits, labels)

        return loss

    def training_step(self, batch: LabelledBatch) -> torch.Tensor:  # type: ignore
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: LabelledBatch, batch_idx: int) -> torch.Tensor:  # type: ignore
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss


class Dataset(torch.utils.data.Dataset):
    def __init__(self, raw_data: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.raw_data = raw_data

    def __len__(self) -> int:
        return len(self.raw_data["input_ids"])

    def __getitem__(self, index):
        return BatchEncoding({k: v[index] for k, v in self.raw_data.items()})


class TrainingDataset(Dataset):
    def __init__(self, raw_data: Dict[str, torch.Tensor], labels: np.ndarray) -> None:
        super().__init__(raw_data)
        self.labels = F.one_hot(torch.tensor(labels).to(torch.long), num_classes=2).to(torch.float)

    def __getitem__(self, index):
        return (
            BatchEncoding({k: v[index] for k, v in self.raw_data.items()}),
            self.labels[index]
        )
