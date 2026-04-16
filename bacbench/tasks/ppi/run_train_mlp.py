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
from torchmetrics.functional import auroc, average_precision

from bacbench.tasks.ppi.data_reader import get_dataloaders_ppi
from bacbench.tasks.utils import get_gpu_info


class PpiLightningModule(pl.LightningModule):
    """Pytorch LightningModule for PPI finetuning."""

    def __init__(self, args, hidden_size: int):
        super().__init__()
        self.args = args
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=args.layer_norm_eps),
            nn.Dropout(0.2),
        )
        self.linear = nn.Linear(hidden_size + 1, 1, bias=True)

        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._test_probs: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

        self.save_hyperparameters()

    def forward(
        self,
        prot1_embeddings: torch.Tensor,
        prot2_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the pairwise PPI head.

        Parameters
        ----------
        prot1_embeddings : torch.Tensor
            Tensor of shape [B, hidden_size] for the first protein in each pair.
        prot2_embeddings : torch.Tensor
            Tensor of shape [B, hidden_size] for the second protein in each pair.
        labels : torch.Tensor
            Binary labels of shape [B].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            BCE loss and per-pair logits.
        """
        # compute cosine similarity
        cosine_sim = torch.cosine_similarity(prot1_embeddings, prot2_embeddings, eps=1e-8).unsqueeze(-1)
        prot1_embeddings = self.dense(self.dropout(prot1_embeddings))
        prot2_embeddings = self.dense(self.dropout(prot2_embeddings))
        pair_embeddings = (prot1_embeddings + prot2_embeddings) / 2.0
        pair_embeddings = torch.cat([pair_embeddings, cosine_sim], dim=-1)

        logits = self.linear(self.dropout(pair_embeddings)).squeeze(-1)
        loss = binary_cross_entropy_with_logits(logits, labels.to(dtype=logits.dtype))
        return loss, logits

    def _log_binary_metrics(self, prefix: str, probs: list[torch.Tensor], labels: list[torch.Tensor]) -> None:
        """Compute epoch-level AUROC/AUPRC from accumulated batch outputs."""
        if not probs:
            return

        probs_tensor = torch.cat(probs).reshape(-1)
        labels_tensor = torch.cat(labels).reshape(-1).long()

        if labels_tensor.unique().numel() < 2:
            metric_auroc = torch.tensor(0.0, device=self.device)
            metric_auprc = torch.tensor(0.0, device=self.device)
        else:
            metric_auroc = auroc(probs_tensor, labels_tensor, task="binary")
            metric_auprc = average_precision(probs_tensor, labels_tensor, task="binary")

        self.log(f"{prefix}_auroc", metric_auroc, prog_bar=(prefix == "val"))
        self.log(f"{prefix}_auprc", metric_auprc, prog_bar=False)

    def training_step(self, batch, batch_idx):
        """Compute the forward pass and log training loss."""
        loss, _ = self.forward(batch["prot1_embeddings"], batch["prot2_embeddings"], batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["labels"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute validation loss and collect outputs for AUROC/AUPRC."""
        loss, logits = self.forward(batch["prot1_embeddings"], batch["prot2_embeddings"], batch["labels"])
        probs = torch.sigmoid(logits.detach())
        labels = batch["labels"].detach().long()
        self._val_probs.append(probs)
        self._val_labels.append(labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=labels.shape[0])
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """Log validation AUROC/AUPRC once per epoch."""
        self._log_binary_metrics("val", self._val_probs, self._val_labels)
        self._val_probs.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        """Compute test loss and collect outputs for AUROC/AUPRC."""
        loss, logits = self.forward(batch["prot1_embeddings"], batch["prot2_embeddings"], batch["labels"])
        probs = torch.sigmoid(logits.detach())
        labels = batch["labels"].detach().long()
        self._test_probs.append(probs)
        self._test_labels.append(labels)
        self.log("test_loss", loss, prog_bar=True, batch_size=labels.shape[0])
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        """Log test AUROC/AUPRC once per epoch."""
        self._log_binary_metrics("test", self._test_probs, self._test_labels)
        self._test_probs.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        """Define the optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)


class ArgumentParser(Tap):
    """Argument parser for PPI MLP training."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_filepath: str
    train_test_split_filepath: str
    output_dir: str

    batch_size: int = 256
    lr: float = 0.001
    layer_norm_eps: float = 1e-12
    weight_decay: float = 0.01

    max_epochs: int = 10
    early_stopping_patience: int = 10
    ckpt_path: str = None
    random_state: int = 30
    max_grad_norm: float = 2.0
    gradient_accumulation_steps: int = 1
    logging_steps: int = 500
    monitor_metric: Literal["loss", "auroc", "auprc"] = "auroc"
    dataloader_num_workers: int = 4

    max_n_proteins: int = 6000
    n_nodes: int = 1
    max_n_ppi_pairs: float = 2 * 1e6
    score_threshold: float = 0.6


def run(args):
    """Main function to run the training."""
    pl.seed_everything(args.random_state)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, "logs"))

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.as_dict(), f)

    train_dl, val_dl, test_dl, hidden_size = get_dataloaders_ppi(
        input_filepath=args.input_filepath,
        train_test_split_filepath=args.train_test_split_filepath,
        max_n_proteins=args.max_n_proteins,
        max_n_ppi_pairs=args.max_n_ppi_pairs,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    if len(train_dl.dataset) == 0:
        raise ValueError("Training split has no usable PPI pairs after preprocessing.")
    if len(val_dl.dataset) == 0:
        raise ValueError("Validation split has no usable PPI pairs after preprocessing.")

    model = PpiLightningModule(args, hidden_size=hidden_size)
    logging.info("Nr of parameters: %d", sum(p.numel() for p in model.parameters()))

    n_gpus, use_ipex = get_gpu_info()
    n_gpus_total = max(n_gpus, 1) * args.n_nodes
    monitor_key = "val_loss" if args.monitor_metric == "loss" else f"val_{args.monitor_metric}"
    monitor_mode = "min" if args.monitor_metric == "loss" else "max"

    early_stop_callback = EarlyStopping(
        monitor=monitor_key,
        patience=args.early_stopping_patience,
        mode=monitor_mode,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best-checkpoint",
        save_top_k=1,
        monitor=monitor_key,
        mode=monitor_mode,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if n_gpus_total > 0 else "cpu",
        devices=n_gpus_total if n_gpus_total > 0 else None,
        precision="bf16" if n_gpus_total > 0 else 32,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[early_stop_callback, checkpoint_callback],
        default_root_dir=args.output_dir,
        enable_checkpointing=True,
        log_every_n_steps=args.logging_steps,
    )

    trainer.fit(model, train_dl, val_dl, ckpt_path=args.ckpt_path)
    val_results = trainer.validate(model=model, dataloaders=val_dl, ckpt_path="best")
    print("Validation metrics:", val_results)

    if test_dl is not None:
        test_results = trainer.test(model=model, dataloaders=test_dl, ckpt_path="best")
        print("Test metrics:", test_results)


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    print(args.as_dict())
    run(args)
