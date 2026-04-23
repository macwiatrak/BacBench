import json
import os
import shutil
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
    "gLM2": 0.001,  # Unknown yet
    "evo": 0.0005,  # Unknown yet
    "ProkBERT": 0.01,  # DONE
    "esm2": 0.01,  # DONE
    "bacformer": 0.01,  # DONE
    "dnabert": 0.001,  # DONE
    "esmc": 0.001,  # DONE
    "mistral_dna": 0.005,  # DONE
    "nucleotide_transformer": 0.001,  # DONE
    "protbert": 0.005,  # DONE
}


def calculate_metrics_per_genome(df: pd.DataFrame):
    """Calculate metrics per genome."""
    gdf = df.groupby("genome_name")[["essential", "logits"]].agg(list).reset_index()
    gdf["auroc"] = gdf.apply(
        lambda x: auroc(
            torch.tensor(x["logits"]), torch.tensor(x["essential"], dtype=torch.long), task="binary", ignore_index=-100
        ).item(),
        axis=1,
    )
    gdf["auprc"] = gdf.apply(
        lambda x: average_precision(
            torch.tensor(x["logits"]), torch.tensor(x["essential"], dtype=torch.long), task="binary", ignore_index=-100
        ).item(),
        axis=1,
    )
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
        return self.net(x).squeeze()

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
            genome_auroc = auroc(genome_preds, genome_labels, task="binary", ignore_index=-100)
            genome_auprc = average_precision(genome_preds, genome_labels, task="binary", ignore_index=-100)
            genome_f1 = f1_score(genome_preds, genome_labels, task="binary", ignore_index=-100)
            output["auroc"].append(genome_auroc.item())
            output["auprc"].append(genome_auprc.item())
            output["f1"].append(genome_f1.item())

        val_auroc = torch.tensor(output["auroc"]).median()
        val_auprc = torch.tensor(output["auprc"]).median()
        val_f1 = torch.tensor(output["f1"]).median()

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
            genome_auroc = auroc(genome_preds, genome_labels, task="binary", ignore_index=-100)
            genome_auprc = average_precision(genome_preds, genome_labels, task="binary", ignore_index=-100)
            genome_f1 = f1_score(genome_preds, genome_labels, task="binary", ignore_index=-100)
            output["auroc"].append(genome_auroc.item())
            output["auprc"].append(genome_auprc.item())
            output["f1"].append(genome_f1.item())

        test_auroc = torch.tensor(output["auroc"]).median()
        test_auprc = torch.tensor(output["auprc"]).median()
        test_f1 = torch.tensor(output["f1"]).median()

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
        persistent_workers=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
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
    input_df_file_path: str = "/Users/maciejwiatrak/Downloads/baclm_masked_hf_inf.parquet"  # "/Users/maciejwiatrak/Downloads/baclm_masked_hf_inf.parquet"
    output_dir: str = "/projects/public/u6fp/benchmarks/tasks/essential-genes/updated/results"
    lr: float = 0.001
    dropout: float = 0.2
    max_epochs: int = 100
    batch_size: int = 256
    num_workers: int = 4
    test: bool = True
    embeddings_col: str = "embeddings"
    model_name: str = None


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    input_dir = "/projects/public/u6fp/benchmarks/tasks/essential-genes/updated/"
    models = [
        # ("dnabert.parquet", "DNABERT-2"),
        # ("bacformer.parquet", "Bacformer"),
        # ("bac_large.parquet", "Bacformer_Large"),
        # ("esmc.parquet", "ESM-C"),  # TOFINISH
        # ("esm2.parquet", "ESM-2"),  # TOFINISH
        ("mistral.parquet", "Mistral-DNA"),  # TOFINISH
        ("nt.parquet", "Nucleotide-Transformer"),  # TOFINISH
        # ("protbert.parquet", "ProtBERT"), # TOFINISH
        # ("glm2.parquet", "gLM2"), # TOFINISH
        # ("prokbert.parquet", "ProkBERT"),
        # ("bac_large_mags.parquet", "Bacformer_Large_MAGS"),
        # ("baclm_masked.parquet", "BacLM-Masked"), # TOFINISH
        # ("baclm_causal.parquet", "BacLM-Causal"), # TOFINISH
        # ("evo2.parquet", "Evo-2"),
        # (
        #     "baclm_with_promoter.parquet",
        #     "BacLM_masked_dna_prot_concat",
        # ),
        # ("evo.parquet", "Evo", "embeddings"),
    ]
    emb_col = args.embeddings_col
    with open("/projects/public/u6fp/benchmarks/tasks/essential-genes/genome_split.json") as f:
        genome_split = json.load(f)

    for model_file, model_name in tqdm(models):
        print(f"Running for model: {model_name}\n\n\n")
        output = []
        df = pd.read_parquet(os.path.join(input_dir, model_file))
        df["split"] = df["genome_name"].map(genome_split)
        best_lr = None
        best_auroc = -1
        os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
        for lr in lrs:
            val_df = main(
                df=df.copy(),
                lr=lr,
                dropout=0.2,
                max_epochs=args.max_epochs,
                batch_size=256,
                num_workers=4,
                output_dir=os.path.join(args.output_dir, model_name),
                random_state=1,
                embeddings_col=emb_col,
                test=False,
            )
            val_auroc_score = val_df["auroc"].median()
            if val_auroc_score > best_auroc:
                best_auroc = val_auroc_score
                best_lr = lr

        for random_state in tqdm([1, 2, 3]):
            test_df = main(
                df=df,
                lr=best_lr,
                dropout=args.dropout,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                output_dir=os.path.join(args.output_dir, model_name),
                random_state=random_state,
                embeddings_col=emb_col,
                test=True,
            )
            test_df["random_state"] = random_state
            output.append(test_df)
        output_df = pd.concat(output)
        output_df["model"] = model_name
        output_df["best_lr"] = best_lr
        output_df.to_parquet(os.path.join(args.output_dir, f"finetune_results_{model_name}.parquet"))
        shutil.rmtree(os.path.join(args.output_dir, model_name))
