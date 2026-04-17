import json
import logging
import os
from typing import Literal

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tap import Tap
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.functional import auroc, average_precision

from bacbench.tasks.ppi.data_reader import get_dataloaders_ppi
from bacbench.tasks.utils import get_gpu_info


def compute_binary_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    """Compute AUROC/AUPRC, returning NaNs when the labels have one class."""
    probs = probs.reshape(-1).float().cpu()
    labels = labels.reshape(-1).long().cpu()

    if labels.numel() == 0 or labels.unique().numel() < 2:
        return float("nan"), float("nan")

    return (
        float(auroc(probs, labels, task="binary").item()),
        float(average_precision(probs, labels, task="binary").item()),
    )


def summarize_test_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pairwise predictions into per-genome metric summaries."""
    if test_df.empty:
        return pd.DataFrame(columns=["genome_name", "label", "probability", "auroc", "auprc"])

    summary_df = test_df.groupby("genome_name", sort=False)[["label", "probability"]].agg(list).reset_index()
    aurocs: list[float] = []
    auprcs: list[float] = []

    for row in summary_df.itertuples(index=False):
        genome_probs = torch.tensor(row.probability, dtype=torch.float32)
        genome_labels = torch.tensor(row.label, dtype=torch.long)
        genome_auroc, genome_auprc = compute_binary_metrics(genome_probs, genome_labels)
        aurocs.append(genome_auroc)
        auprcs.append(genome_auprc)

    summary_df["auroc"] = aurocs
    summary_df["auprc"] = auprcs
    return summary_df


def print_metric_summary(summary_df: pd.DataFrame) -> None:
    """Print aggregate per-genome AUROC/AUPRC statistics."""
    if summary_df.empty:
        print("No test predictions available to summarize.")
        return

    valid_auroc = summary_df["auroc"].dropna()
    valid_auprc = summary_df["auprc"].dropna()
    if valid_auroc.empty or valid_auprc.empty:
        print(f"Per-genome metrics could not be computed for any of the {len(summary_df)} genomes.")
        return

    print(f"Per-genome metrics computed for {len(valid_auroc)}/{len(summary_df)} genomes with both classes present.")
    print(f"AUROC mean: {valid_auroc.mean():.4f}, AUROC std: {valid_auroc.std():.4f}")
    print(f"AUPRC mean: {valid_auprc.mean():.4f}, AUPRC std: {valid_auprc.std():.4f}")
    print(f"AUROC median: {valid_auroc.median():.4f}")
    print(f"AUPRC median: {valid_auprc.median():.4f}")


class PpiLightningModule(pl.LightningModule):
    """Pytorch LightningModule for PPI finetuning."""

    def __init__(
        self,
        hidden_size: int,
        lr: float,
        weight_decay: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.layer_norm_eps = layer_norm_eps

        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=layer_norm_eps),
            nn.Dropout(0.2),
        )
        self.linear = nn.Linear(hidden_size + 1, 1, bias=True)

        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._test_probs: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []
        self._test_genomes: list[str] = []
        self.test_predictions_df_: pd.DataFrame | None = None

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
        metric_auroc_value, metric_auprc_value = compute_binary_metrics(probs_tensor, labels_tensor)
        metric_auroc = torch.tensor(
            0.0 if pd.isna(metric_auroc_value) else metric_auroc_value,
            device=self.device,
        )
        metric_auprc = torch.tensor(
            0.0 if pd.isna(metric_auprc_value) else metric_auprc_value,
            device=self.device,
        )

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

    def on_test_start(self):
        """Reset cached test outputs before evaluation."""
        self._test_probs.clear()
        self._test_labels.clear()
        self._test_genomes.clear()
        self.test_predictions_df_ = None

    def test_step(self, batch, batch_idx):
        """Compute test loss and collect outputs for AUROC/AUPRC."""
        loss, logits = self.forward(batch["prot1_embeddings"], batch["prot2_embeddings"], batch["labels"])
        probs = torch.sigmoid(logits.detach())
        labels = batch["labels"].detach().long()
        self._test_probs.append(probs)
        self._test_labels.append(labels)
        self._test_genomes.extend(batch.get("genome_names", []))
        self.log("test_loss", loss, prog_bar=True, batch_size=labels.shape[0])
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        """Log test AUROC/AUPRC once per epoch and cache pairwise predictions."""
        if not self._test_probs:
            self.test_predictions_df_ = pd.DataFrame(columns=["genome_name", "label", "probability"])
            return

        self._log_binary_metrics("test", self._test_probs, self._test_labels)
        self.test_predictions_df_ = pd.DataFrame(
            {
                "genome_name": self._test_genomes,
                "label": torch.cat(self._test_labels).type(torch.long).cpu().numpy(),
                "probability": torch.cat(self._test_probs).type(torch.float32).cpu().numpy(),
            }
        )
        self._test_probs.clear()
        self._test_labels.clear()
        self._test_genomes.clear()

    def configure_optimizers(self):
        """Define the optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def load_checkpoint_weights(model: PpiLightningModule, checkpoint_path: str) -> None:
    """Load only the model weights from a trusted Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"No 'state_dict' found in checkpoint: {checkpoint_path}")
    model.load_state_dict(state_dict)


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
    early_stopping_patience: int = 3
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

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

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

    model = PpiLightningModule(
        hidden_size=hidden_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        layer_norm_eps=args.layer_norm_eps,
    )
    logging.info("Nr of parameters: %d", sum(p.numel() for p in model.parameters()))

    n_gpus, use_ipex = get_gpu_info()
    n_gpus_total = n_gpus * args.n_nodes if n_gpus > 0 else 0
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
        accelerator="xpu" if use_ipex and n_gpus_total > 0 else ("gpu" if n_gpus_total > 0 else "cpu"),
        devices=n_gpus_total if n_gpus_total > 0 else 1,
        precision="bf16" if n_gpus_total > 0 else 32,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[early_stop_callback, checkpoint_callback],
        default_root_dir=args.output_dir,
        enable_checkpointing=True,
        log_every_n_steps=args.logging_steps,
    )

    trainer.fit(model, train_dl, val_dl, ckpt_path=args.ckpt_path)
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError("Training finished without producing a best checkpoint.")
    load_checkpoint_weights(model, best_model_path)
    logging.info("Loaded best checkpoint weights from %s", best_model_path)

    val_results = trainer.validate(model=model, dataloaders=val_dl, ckpt_path=None)
    print("Validation metrics:", val_results)

    test_results = None
    test_df = None
    if test_dl is not None and len(test_dl.dataset) > 0:
        test_results = trainer.test(model=model, dataloaders=test_dl, ckpt_path=None)
        test_df = model.test_predictions_df_
        if test_df is None:
            raise RuntimeError("Expected test predictions to be cached on the LightningModule after testing.")
        test_summary_df = summarize_test_predictions(test_df)
        print_metric_summary(test_summary_df)
        test_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
        test_summary_df.to_csv(os.path.join(args.output_dir, "test_predictions_by_genome.csv"), index=False)
        print("Test metrics:", test_results)
        return val_results, test_results, test_summary_df
    return val_results, test_results, test_summary_df


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    LRS = [0.1, 0.01, 0.005, 0.001, 0.0005]
    base_output_dir = args.output_dir
    best_df = None
    best_val_auroc = -1.0
    best_lr = None
    lr_results: list[dict[str, float]] = []

    os.makedirs(base_output_dir, exist_ok=True)
    for lr in LRS:
        args.lr = lr
        run_output_dir = os.path.join(base_output_dir, f"lr_{lr}")
        print(f"Running with learning rate: {lr}")
        print({**args.as_dict(), "output_dir": run_output_dir})
        os.makedirs(run_output_dir, exist_ok=True)
        args.output_dir = run_output_dir
        val_results, test_results, test_df = run(args)
        val_auroc = float(val_results[0]["val_auroc"])
        lr_results.append({"lr": lr, "val_auroc": val_auroc})
        if best_val_auroc < val_auroc:
            best_val_auroc = val_auroc
            best_lr = lr
            best_df = test_df
        args.output_dir = base_output_dir

    pd.DataFrame(lr_results).to_csv(os.path.join(base_output_dir, "lr_sweep_results.csv"), index=False)
    if best_df is not None:
        best_df.to_csv(os.path.join(base_output_dir, "best_test_predictions.csv"), index=False)
    if best_lr is not None:
        with open(os.path.join(base_output_dir, "best_lr.json"), "w") as f:
            json.dump({"lr": best_lr, "val_auroc": best_val_auroc}, f, indent=2)
