import json
import logging
import os
from typing import Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tap import Tap
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader

from bacbench.tasks.ppi.data_reader import collate_ppi, get_datasets_ppi
from bacbench.tasks.utils import get_gpu_info


# --------------------------------
# Pytorch LightningModule
# --------------------------------
class PpiLightningModule(pl.LightningModule):
    """Pytorch LightningModule for PPI finetuning."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        # The original head from your code
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.GELU(),
            nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps),
            nn.Dropout(0.2),
        )
        self.linear = nn.Linear(args.hidden_size, 1, bias=True)

        # Save hyperparameters if you want to log them
        self.save_hyperparameters()

    def forward(self, protein_embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward method. It's called inside training/validation/test steps.

        Expects:
          - protein_embeddings of shape [1, N, hidden_size]
          - labels of shape [K, 3], where each row is [idx1, idx2, ppi_label].

        Returns
        -------
          - loss, logits
        """
        # The same aggregator logic you had before
        # (squeeze(0) because batch_size = 1 in your code).
        protein_embeddings = self.dense(self.dropout(protein_embeddings.squeeze(0)))

        # For each row in labels, pick embeddings for idx1 and idx2, then
        # stack them, average them, and feed into linear
        # For demonstration, your original code took the entire set of pairs,
        # cat them, and took the mean. We'll replicate that logic exactly.
        protein_embeddings = torch.cat(
            [protein_embeddings[labels[:, 0]], protein_embeddings[labels[:, 1]]], dim=0
        ).mean(dim=0)

        logits = self.linear(self.dropout(protein_embeddings)).squeeze(-1)

        loss = binary_cross_entropy_with_logits(logits, labels[:, 2].type_as(logits).squeeze(0))
        return loss, logits

    def training_step(self, batch, batch_idx):
        """We compute the forward pass and log the training loss."""
        protein_embeddings = batch.pop("protein_embeddings")
        labels = batch.pop("ppi_labels")
        loss, _ = self.forward(protein_embeddings, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """We compute the forward pass and log the validation loss."""
        protein_embeddings = batch.pop("protein_embeddings")
        labels = batch.pop("ppi_labels")
        loss, logits = self.forward(protein_embeddings, labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.args.batch_size)

        # Return predictions/labels if you want them later in validation_epoch_end
        return {"val_loss": loss, "val_logits": logits.detach(), "val_labels": labels[:, 2].detach()}

    def test_step(self, batch, batch_idx):
        """Test step. Similar to validation step but for test data."""
        protein_embeddings = batch.pop("protein_embeddings")
        labels = batch.pop("ppi_labels")
        loss, logits = self.forward(protein_embeddings, labels)
        self.log("test_loss", loss, prog_bar=True, batch_size=self.args.batch_size)
        return {"test_loss": loss, "test_logits": logits, "test_labels": labels[:, 2]}

    def configure_optimizers(self):
        """Define optimizers and LR schedulers here."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        return optimizer


# --------------------------------
# Argument Parser and run function
# --------------------------------
class ArgumentParser(Tap):
    """Argument parser for training Bacformer (Lightning version)."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str
    output_dir: str

    # model arguments
    batch_size: int = 1
    lr: float = 0.001
    hidden_size: int = 960
    layer_norm_eps: float = 1e-12
    weight_decay: float = 0.01

    # trainer arguments
    max_epochs: int = 10
    early_stopping_patience: int = 10
    ckpt_path: str = None
    random_state: int = 30
    max_grad_norm: float = 2.0
    gradient_accumulation_steps: int = 1
    logging_steps: int = 500
    monitor_metric: Literal["loss", "macro_f1", "macro_accuracy"] = "loss"
    dataloader_num_workers: int = 4

    # data arguments

    max_n_proteins: int = 9000
    n_nodes: int = 1
    max_n_ppi_pairs: float = 3e6
    score_threshold: float = 0.6
    embeddings_col: str = "embeddings"


def run(args):
    """Main function to run the training."""
    pl.seed_everything(args.random_state)

    # Create output dirs if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, "logs"))

    # Save args for reproducibility
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.as_dict(), f)

    # Create our model (LightningModule)
    model = PpiLightningModule(args)
    logging.info("Nr of parameters: %d", sum(p.numel() for p in model.parameters()))

    assert args.batch_size == 1, "Batch size must be 1 for PPI finetuning."

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = get_datasets_ppi(
        input_dir=args.input_dir,
        max_n_proteins=args.max_n_proteins,
        score_threshold=args.score_threshold,
        max_n_ppi_pairs=args.max_n_ppi_pairs,
        embeddings_col=args.embeddings_col,
    )

    n_gpus, use_ipex = get_gpu_info()
    n_gpus_total = max(n_gpus, 1) * args.n_nodes

    # For demonstration, skip replicating 100% of HF logic with warmup steps, etc.
    # but you *can* replicate it in configure_optimizers.

    # If you want mixed precision:
    precision = "bf16" if (n_gpus_total > 0) else 32

    # Create DataLoaders. Your dataset’s __getitem__ should return (embeddings, labels).
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_ppi,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.dataloader_num_workers, collate_fn=collate_ppi
    )

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss" if args.monitor_metric == "loss" else f"val_{args.monitor_metric}",
        patience=args.early_stopping_patience,
        mode="min" if "loss" in args.monitor_metric else "max",
    )

    # ModelCheckpoint to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss" if args.monitor_metric == "loss" else f"val_{args.monitor_metric}",
        mode="min" if "loss" in args.monitor_metric else "max",
    )

    # Create the Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if n_gpus_total > 0 else "cpu",
        devices=n_gpus_total if n_gpus_total > 0 else None,
        precision=precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[early_stop_callback, checkpoint_callback],
        default_root_dir=args.output_dir,
        enable_checkpointing=True,
        # You can set log_every_n_steps=args.logging_steps if desired
        log_every_n_steps=args.logging_steps,
    )

    # Actual training
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)

    # Validate after training (if needed, or to get final metrics)
    val_results = trainer.validate(model, val_loader)
    print("Validation metrics:", val_results)


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    print(args.as_dict())
    run(args)
