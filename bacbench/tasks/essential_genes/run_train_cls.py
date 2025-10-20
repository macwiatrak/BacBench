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
    "gLM2": 0.0005,  # Unknown yet
    "evo": 0.0005,  # Unknown yet
    "ProkBERT": 0.01,  # DONE
    "esm2": 0.01,  # DONE
    "bacformer": 0.001,  # DONE
    "bacformer_large": 0.001,  # DONE
    "dnabert": 0.001,  # DONE
    "esmc": 0.001,  # DONE
    "mistral_dna": 0.005,  # DONE
    "nucleotide_transformer": 0.001,  # DONE
    "protbert": 0.001,  # DONE
}


def calculate_metrics_per_genome(df: pd.DataFrame):
    """Calculate metrics per genome."""
    gdf = df.groupby("genome_name")[["label", "logits"]].agg(list).reset_index()
    gdf["auroc"] = gdf.apply(
        lambda x: auroc(
            torch.tensor(x["logits"]), torch.tensor(x["label"], dtype=torch.long), task="binary", ignore_index=-100
        ).item(),
        axis=1,
    )
    gdf["auprc"] = gdf.apply(
        lambda x: average_precision(
            torch.tensor(x["logits"]), torch.tensor(x["label"], dtype=torch.long), task="binary", ignore_index=-100
        ).item(),
        axis=1,
    )
    print("Mean AUROC:", gdf["auroc"].mean(), "Median AUROC:", gdf["auroc"].median())
    print("Mean AUPRC:", gdf["auprc"].mean(), "Median AUPRC:", gdf["auprc"].median())
    return df


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


def prepare_essential_genes_df(df: pd.DataFrame, embeddings_col: str) -> pd.DataFrame:
    """Prepare the essential genes DataFrame."""
    # check if the embeddings column is already in the correct format
    if not isinstance(df[embeddings_col].iloc[0][0], np.ndarray):
        return df
    # convert list of lists to just list as there is only one contig per genome
    df[embeddings_col] = df[embeddings_col].apply(lambda x: x[0])
    df[embeddings_col] = df[embeddings_col].apply(lambda x: x[0])
    # explode the DF
    df = df.explode([embeddings_col, "essential", "protein_id", "product", "start", "end"])
    return df


def main(
    input_df_dile_path: str,
    lr: float = 1e-3,
    dropout: float = 0.2,
    max_epochs: int = 100,
    batch_size: int = 512,
    num_workers: int = 4,
    output_dir: str = "/tmp/evo-output/",
    random_state: int = 1,
    embeddings_col: str = "embeddings",
    # genome2split_file_path: str | None = None,
    test: bool = False,
):
    """Run the training of the Linear model."""
    # set the random seed
    set_seed(random_state)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read input file
    df = pd.read_parquet(input_df_dile_path)

    # add the split column given the genome2split mapping if it exists
    # if genome2split_file_path is not None:
    #     with open(genome2split_file_path, "r") as f:
    #         genome2split = json.load(f)
    #     df['split'] = df['genome_name'].map(genome2split)

    # explode the embeddings column as after embedding it is a list of lists
    df = prepare_essential_genes_df(df, embeddings_col=embeddings_col)
    # process the DF
    genome2idx = {g: i for i, g in enumerate(df["genome_name"].unique())}
    df["genome_idx"] = df["genome_name"].map(genome2idx)
    df["label"] = df.essential.map({"Yes": 1, "No": 0})
    dim = df[embeddings_col].iloc[0].shape[0]

    # split the data
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "val"]
    if len(test_df) == 0:
        test_df = df[df["split"] == "validation"]
    val_df = df[df["split"] == "test"]

    # create datasets
    train_dataset = TensorDataset(
        torch.tensor(np.stack(train_df[embeddings_col].tolist()), dtype=torch.float32),
        torch.tensor(train_df.label.tolist(), dtype=torch.long),
        torch.tensor(train_df.genome_idx.tolist(), dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(np.stack(val_df[embeddings_col].tolist()), dtype=torch.float32),
        torch.tensor(val_df.label.tolist(), dtype=torch.long),
        torch.tensor(val_df.genome_idx.tolist(), dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(np.stack(test_df[embeddings_col].tolist()), dtype=torch.float32),
        torch.tensor(test_df.label.tolist(), dtype=torch.long),
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

    if not test:
        return

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
    _ = calculate_metrics_per_genome(val_df)

    print("Test metrics:")
    trainer.test(model, test_dataloader)
    output = []
    with torch.no_grad():
        for batch in test_dataloader:
            output.append(model(batch[0]))
    test_df["logits"] = torch.cat(output).cpu().numpy()
    test_df = calculate_metrics_per_genome(test_df)
    test_df = test_df.drop(columns=[embeddings_col])
    return test_df


class ArgumentParser(Tap):
    """Argument parser for finetuning linear model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_df_file_path: str = "/Users/maciejwiatrak/Downloads/evo_embeds_all.parquet"
    output_dir: str = "/tmp/eg-ft"
    lr: float = 0.0005
    dropout: float = 0.2
    max_epochs: int = 100
    batch_size: int = 256
    num_workers: int = 4
    test: bool = True
    embeddings_col: str = "gene_embedding"
    model_name: str = "evo"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    output = []
    for random_state in tqdm([1, 2, 3]):
        test_df = main(
            input_df_dile_path=args.input_df_file_path,
            lr=args.lr,
            # genome2split_file_path=args.genome2split_file_path,
            dropout=args.dropout,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            random_state=random_state,
            embeddings_col=args.embeddings_col,
            test=args.test,
        )
        test_df["random_state"] = random_state
        output.append(test_df)
    output_df = pd.concat(output)
    output_df.to_parquet(os.path.join(args.output_dir, f"finetune_results_{args.model_name}.parquet"))
