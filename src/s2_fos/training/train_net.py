import argparse
import pandas as pd
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
import wandb
import pytorch_lightning.callbacks as pl_callbacks
from typing import Optional, List
from datasets import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from torch.optim import AdamW
import os
import json

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
"""
['Agricultural and Food sciences', 'Art', 'Biology', 'Business',
       'Chemistry', 'Computer science', 'Economics', 'Education',
       'Engineering', 'Environmental science', 'Geography', 'Geology',
       'History', 'Law', 'Linguistics', 'Materials science',
       'Mathematics', 'Medicine', 'Philosophy', 'Physics',
       'Political science', 'Psychology', 'Sociology']
"""

class GradientsLogging(pl_callbacks.Callback):
    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % trainer.log_every_n_steps == 0:
            for name, params in pl_module.named_parameters():
                if params.requires_grad and params.grad is not None:
                    trainer.logger.experiment.log(
                        {f"gradients/{name}": wandb.Histogram(params.grad.detach().cpu())},
                        step=trainer.global_step,
                    )

class Hill(nn.Module):
    r"""Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "
    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2
    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter,
          : math:`\gamma` is a commonly used value same as Focal loss.
    .. note::
        Sigmoid will be done in loss.
    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2
    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward
        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt**self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1 - targets) * -(self.lamb - pred_neg) * pred_neg**2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg, self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.mean()

class MultiLabelDataModule(LightningDataModule):
    """Pytorch Lightning Data Module for MultiLabel classification."""

    loader_columns = [
        "sID",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_fields: List[str],
        model_name_or_path: str = "allenai/scibert_scivocab_uncased",
        max_seq_length: int = 128,
        batch_size: int = 16,
        eval_batch_size: int = 512,
        num_workers: int = 4,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.text_fields = text_fields
        self.num_labels = 23
        self.shuffle = shuffle
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = {
            "train": Dataset.from_pandas(self.train_df),
            "val": Dataset.from_pandas(self.val_df),
            "test": Dataset.from_pandas(self.test_df),
        }

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

    def train_dataloader(self, shuffle=None):
        if shuffle is None:
            shuffle = self.shuffle
        return DataLoader(
            self.dataset["train"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["val"], batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False
        )

    def convert_to_features(self, example_batch):
        # Concatenate text fields for each example
        # Concatenate text fields for each example
        num_examples = len(example_batch[self.text_fields[0]])
        # For bert put seperated tokens from tokenizer tokenizer.sep_token
        text_examples = [
            f'{self.tokenizer.sep_token}'.join([(example_batch[field][i] if example_batch[field][i] is not None else '') for field in self.text_fields]) for i in range(num_examples)
        ]

        features = self.tokenizer.batch_encode_plus(
            text_examples,
            max_length=self.max_seq_length,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        return features


class MultiLabelTransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        hidden_dropout_prob: float = 0.1,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        label_names: Optional[list] = None,
        metrics=None,
        loss=None,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_hyperparameters()
        if label_names is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        else:
            id2label = {i: label_name for i, label_name in enumerate(label_names)}
            label2id = {val: key for key, val in id2label.items()}
            self.config = AutoConfig.from_pretrained(
                model_name_or_path, num_labels=num_labels, id2label=id2label, label2id=label2id
            )
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

        if loss is None:
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        else:
            self.loss = loss

        # accuracy and F1 cause weird errors...
        if metrics is None:
            self.metrics = [
                ("auroc", MultilabelAUROC(num_labels=self.num_labels)),
                ("average_precision", MultilabelAveragePrecision(num_labels=self.num_labels)),
            ]
        else:
            self.metrics = []

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs

    def _process_batch(self, batch, batch_idx, phase="train"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # Convert labels to tensor and move to the same device as the model
        labels = torch.tensor(batch["labels"], device=self.device, dtype=torch.float)
        outputs = self(input_ids, attention_mask).logits
        loss = self.loss(outputs, labels)
        labels_int = labels.to(torch.int)  # Update this line as well

        self.log(f"{phase}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric in self.metrics:
            loss_in_the_loop = metric(outputs, labels_int)
            self.log(
                f"{phase}_{metric_name}", loss_in_the_loop, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self._process_batch(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._process_batch(batch, batch_idx, "val")["loss"]

    def test_step(self, batch, batch_idx):
        return self._process_batch(batch, batch_idx, "test")["loss"]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self(input_ids, attention_mask)
        return outputs

    def setup(self, stage=None):
        if stage != "fit":
            return

        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = train_loader.batch_size * max(1, self.trainer.num_devices)
        self.total_steps = (
            float(self.trainer.max_epochs)
            * (len(train_loader.dataset) // tb_size)
            // self.trainer.accumulate_grad_batches
        )

    def configure_optimizers(self):
        """Prepare optimizer and schedule (cosine warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}. Currently only supporting adamw.")

        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.warmup_ratio * self.total_steps),
                num_training_steps=self.total_steps,
            )
        elif self.scheduler == "warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.warmup_ratio * self.total_steps),
            )
        elif self.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.warmup_ratio * self.total_steps),
                num_training_steps=self.total_steps,
            )
        elif self.scheduler == "constant":
            scheduler = get_constant_schedule(optimizer)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    def validate_loss(value):
        if value not in ['Hill', 'Asymmetric', None]:
            raise argparse.ArgumentTypeError(f"Invalid loss function. Expected one of: {value}")
        return value


    def check_hill_loss(value):
        value = float(value)
        if value < 0 or value > 3:
            raise argparse.ArgumentTypeError(
                "%s is an invalid value for Hill loss parameters. It should be in the range [0, 3]." % value)
        return value


    def check_asymmetric_loss(value):
        value = float(value)
        if value < 0 or value > 5:
            raise argparse.ArgumentTypeError(
                "%s is an invalid value for Asymmetric loss parameters. It should be in the range [0, 5]." % value)
        return value

    parser = argparse.ArgumentParser(description="Multi-label classification with transformers")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the train data parquet file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data parquet file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data parquet file")
    parser.add_argument("--text_fields", type=str, nargs="+", default=['title', 'normalized_venue_name', 'abstract'], help="List of input text fields")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--train", type=str2bool, required=True, default=True, help="Run training or test evaluation only?")
    parser.add_argument("--model_checkpoint_path", type=str, required=False, default=None, help="Path to model checkpoints")
    parser.add_argument("--project_name", type=str, required=True, help="sciBert-finetune")
    parser.add_argument("--batch_size", type=int, required=False, default=16, help="batch_size")
    parser.add_argument("--learning_rate", type=float, required=False, default=1e-5, help="learning_rate")
    parser.add_argument("--warmup_ratio", type=float, required=False, default=0.06, help="warmup_ratio")
    parser.add_argument("--wandb_name", type=str, required=False, default='scibert_finetune_fos', help="")
    parser.add_argument("--wandb_run_des", type=str, required=False, default="", help="")
    parser.add_argument("--log_dir", type=str, required=False, default="/output", help="library where the logs are outputed to")
    parser.add_argument("--loss", type=validate_loss, required=False, default=None, help="Name of the loss used for training possible values: Hill, Asymetric")

    # Hill loss parameters
    parser.add_argument("--hill_lamb", type=check_hill_loss, default=1.5)
    parser.add_argument("--hill_margin", type=check_hill_loss, default=1.0)
    parser.add_argument("--hill_gamma", type=check_hill_loss, default=2.0)
    parser.add_argument("--hill_reduction", type=str, default="mean", choices=["mean", "sum"])

    # Asymmetric loss parameters
    parser.add_argument("--asym_gamma_neg", type=check_asymmetric_loss, default=4)
    parser.add_argument("--asym_gamma_pos", type=check_asymmetric_loss, default=1)
    parser.add_argument("--asym_clip", type=check_asymmetric_loss, default=0.05)
    parser.add_argument("--asym_eps", type=float, default=1e-8)
    parser.add_argument("--asym_disable_torch_grad_focal_loss", type=bool, default=False)

    args = parser.parse_args()

    hill_loss_params = {
        "lamb": args.hill_lamb,
        "margin": args.hill_margin,
        "gamma": args.hill_gamma,
        "reduction": args.hill_reduction
    }

    asymmetric_loss_params = {
        "gamma_neg": args.asym_gamma_neg,
        "gamma_pos": args.asym_gamma_pos,
        "clip": args.asym_clip,
        "eps": args.asym_eps,
        "disable_torch_grad_focal_loss": args.asym_disable_torch_grad_focal_loss
    }
    args = parser.parse_args()

    train_data = pd.read_parquet(args.train_data)
    test_data = pd.read_parquet(args.test_data)
    val_data = pd.read_parquet(args.val_data)

    wandb_logger = WandbLogger(project=args.project_name, entity="egork", name=args.wandb_name, notes=args.wandb_run_des)
    data_module = MultiLabelDataModule(
        train_df=train_data,
        val_df=val_data,
        test_df=test_data,
        text_fields=args.text_fields,
        model_name_or_path='allenai/scibert_scivocab_uncased',
        max_seq_length=128,
        batch_size=args.batch_size,
        eval_batch_size=512,
        num_workers=4
    )
    hill_loss = Hill(**hill_loss_params)
    asymmetric_loss = AsymmetricLoss(**asymmetric_loss_params)

    if args.loss.lower() == "hill":
        loss = hill_loss
    elif args.loss.lower() == "asymmetric":
        loss = asymmetric_loss
    else:
        loss = None

    print(f'Using loss: {args.loss}')
    print(f'With parameters: {hill_loss_params if args.loss.lower() == "hill" else asymmetric_loss_params}')

    # Create a MultiLabelTransformer instance
    model = MultiLabelTransformer(
        model_name_or_path='allenai/scibert_scivocab_uncased',
        num_labels=data_module.num_labels,
        learning_rate=args.learning_rate,
        adam_epsilon=1e-8,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.001,
        loss=loss
    )

    # Create a ModelCheckpoint callback
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=args.save_path,
        filename="best-model-hyer-parameter-search-{epoch}",
        save_top_k=1,  # Save a checkpoint at each epoch
        verbose=True,
        monitor="val_average_precision",
        mode="max",
    )

    early_stop_callback = pl_callbacks.EarlyStopping(
        monitor="val_average_precision",
        min_delta=0.00,  # Minimum change in the monitored quantity (optional)
        patience=1,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,  # Whether to print logs or not
        mode="max",  # "max" is for maximizing the metric and "min" for minimizing the metric
    )

    log_dir = "./"
    csv_logger = CSVLogger(log_dir)

    callbacks = [checkpoint_callback, early_stop_callback, GradientsLogging()]

    # Check if GPU is available
    gpus = 1 if torch.cuda.is_available() else 0
    # Train the model using PyTorch Lightning's Trainer with the ModelCheckpoint callback
    trainer = Trainer(accelerator='gpu' if gpus else 'cpu', devices=1, max_epochs=5, callbacks=callbacks, logger=[wandb_logger,csv_logger])
    if args.train:
      trainer.fit(model, data_module)
      trainer.test(model, datamodule=data_module)
    else:
      model_checkpoint_path = args.model_checkpoint_path
      # List all files in the model_checkpoint_path
      all_files = os.listdir(model_checkpoint_path)
      # Filter the files based on your criteria, e.g., file extension
      # Replace '.pt' with the desired file extension for your model files
      model_files = [file for file in all_files if file.endswith('.ckpt')]
      for model_name in model_files:
        wandb_logger.experiment.name = model_name
        wandb_logger.experiment.save()  # Save the updated run name
        model = MultiLabelTransformer.load_from_checkpoint(os.path.join(model_checkpoint_path, model_name))
        trainer.test(model, datamodule=data_module)