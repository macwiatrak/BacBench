import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tap import Tap
from torch.optim import AdamW
from torch.utils.data import TensorDataset
from torchmetrics.functional import auroc, average_precision, f1_score
from tqdm import tqdm
from transformers import set_seed

# learnigng rates for different models after tuning on the validation set
MODEL2LR = {
    "gLM2": 0.01,
    "Evo": 0.1,
    "Evo-2": 0.05,
    "ProkBERT": 0.01,
    "ESM-2": 0.001,
    "Bacformer": 0.01,
    "Bacformer_Large": 0.1,
    "BacLM": 0.001,
    "DNABERT-2": 0.001,
    "ESM-C": 0.005,
    "Mistral-DNA": 0.01,
    "Nucleotide_Transformer": 0.001,
    "ProtBERT": 0.0001,
}


def _has_both_binary_classes(labels: torch.Tensor) -> bool:
    """Return True when labels contain both binary classes after dropping ignored targets."""
    valid_labels = labels[labels != -100]
    return torch.unique(valid_labels).numel() >= 2


def _median_or_zero(values: list[float]) -> torch.Tensor:
    """Return the median of values or 0.0 when the list is empty."""
    if len(values) == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32).median()


def calculate_metrics_per_genome(df: pd.DataFrame):
    """Calculate metrics per genome."""
    gdf = df.groupby("genome_name")[["essential", "logits"]].agg(list).reset_index()

    def _safe_ranking_metrics(row: pd.Series) -> pd.Series:
        logits = torch.tensor(row["logits"])
        labels = torch.tensor(row["essential"], dtype=torch.long)
        if not _has_both_binary_classes(labels):
            return pd.Series({"auroc": np.nan, "auprc": np.nan})
        return pd.Series(
            {
                "auroc": auroc(logits, labels, task="binary", ignore_index=-100).item(),
                "auprc": average_precision(logits, labels, task="binary", ignore_index=-100).item(),
            }
        )

    gdf[["auroc", "auprc"]] = gdf.apply(_safe_ranking_metrics, axis=1)
    print("Mean AUROC:", gdf["auroc"].mean(), "Median AUROC:", gdf["auroc"].median())
    print("Mean AUPRC:", gdf["auprc"].mean(), "Median AUPRC:", gdf["auprc"].median())
    return gdf


class LinearModel(pl.LightningModule):
    """PyTorch Lightning Linear model for finetuning."""

    def __init__(self, dim: int = 8192, dropout: float = 0.2, lr: float = 1e-3):
        """Initialize the model"""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lr = lr
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),  # <— always 1 output
        )
        self.save_hyperparameters(logger=False)

        # Buffers to store predictions/labels across an epoch
        self.val_preds = []
        self.val_labels = []
        self.val_genome_indices = []
        self.test_preds = []
        self.test_labels = []
        self.test_genome_indices = []

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        x = self.dropout(x)
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y, genome_idx = batch
        preds = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(preds, y.type_as(preds))
        return loss

    # 1) Validation
    def on_validation_start(self):
        """Clear buffers at the start of validation."""
        self.val_preds.clear()
        self.val_labels.clear()
        self.val_genome_indices.clear()

    def validation_step(self, batch, batch_idx):
        """Accumulate predictions/labels in each validation step."""
        x, y, genome_idx = batch
        preds = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(preds, y.type_as(preds))

        # Collect preds and labels
        self.val_preds.append(preds.detach())  # optionally .cpu()
        self.val_labels.append(y.detach())  # optionally .cpu()
        self.val_genome_indices.append(genome_idx.detach())

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """Compute metrics on all predictions and labels."""
        # Concatenate over the entire validation set
        all_preds = torch.cat(self.val_preds, dim=0)
        all_labels = torch.cat(self.val_labels, dim=0)
        all_genome_indices = torch.cat(self.val_genome_indices, dim=0)

        # Compute metrics
        val_loss = F.binary_cross_entropy_with_logits(all_preds, all_labels.type_as(all_preds))

        # compute metrics per genome
        output = defaultdict(list)
        for genome_idx in all_genome_indices.unique():
            idxs = all_genome_indices == genome_idx
            genome_preds = all_preds[idxs]
            genome_labels = all_labels[idxs]
            genome_f1 = f1_score(genome_preds, genome_labels, task="binary", ignore_index=-100)
            output["f1"].append(genome_f1.item())
            if _has_both_binary_classes(genome_labels):
                genome_auroc = auroc(genome_preds, genome_labels, task="binary", ignore_index=-100)
                genome_auprc = average_precision(genome_preds, genome_labels, task="binary", ignore_index=-100)
                output["auroc"].append(genome_auroc.item())
                output["auprc"].append(genome_auprc.item())

        val_auroc = _median_or_zero(output["auroc"])
        val_auprc = _median_or_zero(output["auprc"])
        val_f1 = _median_or_zero(output["f1"])

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_auroc", val_auroc, prog_bar=True)
        self.log("val_auprc", val_auprc, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)

    #
    # 2) Testing
    #
    def on_test_start(self):
        """Clear buffers at the start of test."""
        self.test_preds.clear()
        self.test_labels.clear()
        self.test_genome_indices.clear()

    def test_step(self, batch, batch_idx):
        """Accumulate predictions/labels in each test step."""
        x, y, genome_idx = batch
        preds = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(preds, y.type_as(preds))

        # Collect preds and labels
        self.test_preds.append(preds.detach())  # optionally .cpu()
        self.test_labels.append(y.detach())  # optionally .cpu()
        self.test_genome_indices.append(genome_idx.detach())

        return {"test_loss": loss}

    def on_test_epoch_end(self):
        """Compute metrics on all predictions and labels."""
        all_preds = torch.cat(self.test_preds, dim=0)
        all_labels = torch.cat(self.test_labels, dim=0)
        all_genome_indices = torch.cat(self.test_genome_indices, dim=0)

        # Compute metrics
        test_loss = F.binary_cross_entropy_with_logits(all_preds, all_labels.type_as(all_preds))

        # compute metrics per genome
        output = defaultdict(list)
        for genome_idx in all_genome_indices.unique():
            idxs = all_genome_indices == genome_idx
            genome_preds = all_preds[idxs]
            genome_labels = all_labels[idxs]
            genome_f1 = f1_score(genome_preds, genome_labels, task="binary", ignore_index=-100)
            output["f1"].append(genome_f1.item())
            if _has_both_binary_classes(genome_labels):
                genome_auroc = auroc(genome_preds, genome_labels, task="binary", ignore_index=-100)
                genome_auprc = average_precision(genome_preds, genome_labels, task="binary", ignore_index=-100)
                output["auroc"].append(genome_auroc.item())
                output["auprc"].append(genome_auprc.item())

        test_auroc = _median_or_zero(output["auroc"])
        test_auprc = _median_or_zero(output["auprc"])
        test_f1 = _median_or_zero(output["f1"])

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_auroc", test_auroc, prog_bar=True)
        self.log("test_auprc", test_auprc, prog_bar=True)
        self.log("test_f1", test_f1, prog_bar=True)

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = AdamW(
            params=[p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=0.02,
        )
        return optimizer


def main(
    df: pd.DataFrame,
    lr: float = 1e-3,
    dropout: float = 0.2,
    max_epochs: int = 100,
    batch_size: int = 512,
    num_workers: int = 4,
    output_dir: str = "/tmp/evo-output/",
    random_state: int = 1,
    embeddings_col: str = "embeddings",
    test: bool = False,
):
    """Run the training of the Linear model."""
    # set the random seed
    set_seed(random_state)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # explode the pandas dataframe so that each row corresponds to one gene
    df = df.explode([embeddings_col, "essential"]).explode([embeddings_col, "essential"])

    genome2idx = {g: i for i, g in enumerate(df["genome_name"].unique())}
    df["genome_idx"] = df["genome_name"].map(genome2idx)
    dim = df[embeddings_col].iloc[0].shape[0]

    # split the data
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    # create datasets
    train_dataset = TensorDataset(
        torch.tensor(np.stack(train_df[embeddings_col].tolist()), dtype=torch.float32),
        torch.tensor(train_df.essential.tolist(), dtype=torch.long),
        torch.tensor(train_df.genome_idx.tolist(), dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(np.stack(val_df[embeddings_col].tolist()), dtype=torch.float32),
        torch.tensor(val_df.essential.tolist(), dtype=torch.long),
        torch.tensor(val_df.genome_idx.tolist(), dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(np.stack(test_df[embeddings_col].tolist()), dtype=torch.float32),
        torch.tensor(test_df.essential.tolist(), dtype=torch.long),
        torch.tensor(test_df.genome_idx.tolist(), dtype=torch.long),
    )

    # create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # create the model
    model = LinearModel(lr=lr, dropout=dropout, dim=dim)

    # create the trainer
    early_stop_callback = EarlyStopping(
        monitor="val_auroc",
        patience=10,
        verbose=True,
        mode="max",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="best-{epoch:02d}-{val_auroc:.4f}",
        monitor="val_auroc",
        save_top_k=1,
        save_last=True,
        mode="max",
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu" if not torch.cuda.is_available() else "auto",
        devices="auto",
        enable_checkpointing=True,
        enable_model_summary=True,
        # deterministic=True,
        default_root_dir=output_dir,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
        ],
    )

    # train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    model = LinearModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    print("Best model path:", trainer.checkpoint_callback.best_model_path)
    model.eval()

    print("Val metrics:")
    trainer.test(model, val_dataloader)
    output = []
    with torch.no_grad():
        for batch in val_dataloader:
            output.append(model(batch[0]))
    val_df["logits"] = torch.cat(output).cpu().numpy()
    val_df = calculate_metrics_per_genome(val_df)

    if not test:
        return val_df

    print("Test metrics:")
    trainer.test(model, test_dataloader)
    output = []
    with torch.no_grad():
        for batch in test_dataloader:
            output.append(model(batch[0]))
    test_df["logits"] = torch.cat(output).cpu().numpy()
    test_df = calculate_metrics_per_genome(test_df)
    return test_df


class ArgumentParser(Tap):
    """Argument parser for finetuning linear model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_df_file_path: str
    output_dir: str
    lr: float
    dropout: float = 0.2
    max_epochs: int = 100
    batch_size: int = 256
    num_workers: int = 4
    test: bool = True
    embeddings_col: str = "embeddings"
    model_name: str = None


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    df = pd.read_parquet(args.input_df_file_path)
    # train and test the model for different random seeds to get a distribution of results
    output = []
    for random_state in tqdm([1, 2, 3]):
        test_df = main(
            df=df,
            lr=args.lr,
            dropout=args.dropout,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            random_state=random_state,
            embeddings_col=args.embeddings_col,
            test=True,
        )
        test_df["random_state"] = random_state
        output.append(test_df)
    output_df = pd.concat(output)
    output_df["model"] = args.model_name
    output_df.to_parquet(os.path.join(args.output_dir, f"finetune_results_{args.model_name}.parquet"))
