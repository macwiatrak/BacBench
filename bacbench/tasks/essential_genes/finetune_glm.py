import os
from functools import partial

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # PL ≥2.0
from sklearn.metrics import average_precision_score, roc_auc_score
from tap import Tap
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from transformers import AutoModel, AutoTokenizer


# ------------------------- data helpers --------------------------------
class gLM2Dataset(Dataset):
    """
    Wrap a pandas DataFrame that has at minimum the columns
    ─────────────────────────────────────────────────────────
    genome_name : str   – key for the `seqs` dict
    strand      : str   – “+” or “-”
    start       : int   – CDS start (0‑based, inclusive)
    end         : int   – CDS end   (0‑based, exclusive)
    sequence    : str   – protein sequence (AA letters, upper‑case)
    label       : …     – any label you need
    ─────────────────────────────────────────────────────────
    `seqs` is a dict: {genome_name → full genomic DNA string (lower‑ or upper‑case)}.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seqs: dict[str, str],
        promoter_len: int,
        max_seq_len: int = 4_096,
    ):
        self.df = df.reset_index(drop=True)
        self.seqs = seqs
        self.promoter_len = promoter_len
        self.max_seq_len = max_seq_len

    # ---------- helpers -------------------------------------------------
    @staticmethod
    def _rev_comp(seq: str) -> str:
        """Reverse‑complement a DNA string."""
        tbl = str.maketrans("acgtACGT", "tgcaTGCA")
        return seq.translate(tbl)[::-1]

    # ---------- standard Dataset API -----------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        dna = self.seqs[row.genome_name]  # full genomic DNA
        start = int(row.start)
        end = int(row.end)
        strand = row.strand
        protein = str(row.sequence).upper()  # upper‑case AA tokens

        # --- 1. extract promoter (up‑stream on +, down‑stream on –) -----
        if strand == "+":
            # promoter is upstream of start
            prom_start = max(0, start - self.promoter_len)
            promoter = dna[prom_start:start].lower()
            strand_tok = "<+>"
            seq = f"<+>{promoter}{strand_tok}{protein}"
        else:  # strand == "‑" (reverse)
            # promoter is downstream of gene end → reverse‑complement it
            prom_end = min(len(dna), end + self.promoter_len)
            promoter = self._rev_comp(dna[end:prom_end]).lower()
            strand_tok = "<->"
            seq = f"{strand_tok}{protein}<+>{promoter}"

        # hard cut if caller asked for a shorter window
        seq = seq[: self.max_seq_len]

        return {"sequence": seq, "label": row.label}


def add_strand_info(df: pd.DataFrame, dna_df: pd.DataFrame) -> pd.DataFrame:
    """Add strand info"""
    strand_df = dna_df[["genome_name", "strand", "start", "protein_id"]].explode(["strand", "start", "protein_id"])
    df = pd.merge(df, strand_df, on=["genome_name", "start", "protein_id"], how="left")
    df["strand"] = df["strand"].fillna(1.0)
    return df


def collate_prots(tokenizer, max_seq_len, batch):
    """Pad to the longest sequence in *this* batch (length‑sorted data)."""
    seqs = [b["sequence"] for b in batch]
    inputs = tokenizer(
        seqs,
        add_special_tokens=True,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_seq_len,
    )
    inputs["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return inputs


# ------------------------- Lightning module ----------------------------
class PlmEssentialGeneClassifier(pl.LightningModule):
    """Finetune essential gene classifier on protein sequences."""

    def __init__(self, model, hidden_size: int, lr: float = 1e-5, dropout: float = 0.2, model_type: str = "esm2"):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.lr = lr

        self.model = model
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.model_type = model_type

        self.criterion = nn.BCEWithLogitsLoss()
        self.val_auc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_probs = []

    # ------------- forward & common step helpers -----------------------
    def forward(self, inputs):
        """Forward pass through the model."""
        last_hidden_state = self.model(inputs["input_ids"], output_hidden_states=True).last_hidden_state
        out = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        logits = self.classifier(self.dropout(out)).squeeze()  # (B,)
        return logits

    def _shared_step(self, batch, stage):
        y_hat = self(batch)
        y = batch["labels"]
        loss = self.criterion(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=y.size(0))
        return y_hat, y, loss

    # ------------- PL step hooks ---------------------------------------
    def training_step(self, batch, _):
        """Training step for Lightning."""
        return self._shared_step(batch, "train")[-1]

    def validation_step(self, batch, _):
        """Validation step for Lightning."""
        y_hat, y, loss = self._shared_step(batch, "val")
        probs = torch.sigmoid(y_hat)
        self.val_auc.update(probs, y.int())
        self.val_auprc.update(probs, y.int())
        return loss

    def on_validation_epoch_end(self):
        """Log validation metrics at the end of the epoch."""
        auc = self.val_auc.compute()
        auprc = self.val_auprc.compute()
        self.log_dict({"val_auc": auc, "val_auprc": auprc}, prog_bar=True, sync_dist=True)
        self.val_auc.reset()
        self.val_auprc.reset()

    def test_step(self, batch):
        """Test step for Lightning."""
        logits = self(batch)
        self.test_probs.append(torch.sigmoid(logits).cpu())

    def on_test_epoch_end(self):
        """Log test predictions at the end of the epoch."""
        self.test_probs = torch.cat(self.test_probs).type(torch.float32).numpy()

    # ------------- optimiser ------------------------------------------
    def configure_optimizers(self):
        """Configure optimizer for Lightning."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ------------------------- main run() ----------------------------------
def run(
    model_path: str,
    output_dir: str,
    hidden_size: int,
    lr: float = 1e-5,
    dropout: float = 0.2,
    batch_size: int = 64,
    max_seq_len: int = 1024,
    num_epochs: int = 10,
    gradient_accumulation_steps: int = 8,
    promoter_len: int = 128,
    prot_dataset_path: str = "macwiatrak/bacbench-essential-genes-protein-sequences",
    dna_dataset_path: str = "macwiatrak/bacbench-essential-genes-dna",
):
    """Finetune a pretrained protein language model on essential gene classification."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # 0) prepare data (already sorted for efficiency)
    ds = load_dataset(prot_dataset_path)
    dna_ds = load_dataset(dna_dataset_path)

    def _prep(split):
        df = ds[split].to_pandas().explode(["protein_id", "product", "start", "end", "essential", "sequence"])
        df["label"] = df.essential.map({"Yes": 1, "No": 0})
        return df.sort_values("sequence", key=lambda x: x.str.len(), ascending=False)

    train_df, val_df, test_df = map(_prep, ["train", "validation", "test"])
    # get DNA sequences for each genome
    train_dna_df = dna_ds["train"].to_pandas()
    train_df = add_strand_info(train_df, train_dna_df)
    train_seqs = {row.genome_name: row.dna_seq for _, row in train_dna_df.iterrows()}
    val_dna_df = dna_ds["validation"].to_pandas()
    val_df = add_strand_info(val_df, val_dna_df)
    val_seqs = {row.genome_name: row.dna_seq for _, row in val_dna_df.iterrows()}
    test_dna_df = dna_ds["test"].to_pandas()
    test_df = add_strand_info(test_df, test_dna_df)
    test_seqs = {row.genome_name: row.dna_seq for _, row in test_dna_df.iterrows()}

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.train()

    # 2) datasets & dataloaders
    train_ds = gLM2Dataset(train_df, train_seqs, promoter_len, max_seq_len)
    val_ds = gLM2Dataset(val_df, val_seqs, promoter_len, max_seq_len)
    test_ds = gLM2Dataset(test_df, test_seqs, promoter_len, max_seq_len)

    collate_fn = partial(collate_prots, tokenizer, max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3) Lightning objects
    model = PlmEssentialGeneClassifier(model=model, hidden_size=hidden_size, lr=lr, dropout=dropout)

    ckpt_cb = ModelCheckpoint(dirpath=output_dir, monitor="val_auc", mode="max", save_top_k=1, filename="best-val_auc")
    early_cb = EarlyStopping(monitor="val_auc", mode="max", patience=5)  # :contentReference[oaicite:8]{index=8}

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accumulate_grad_batches=gradient_accumulation_steps,
        callbacks=[ckpt_cb, early_cb],
        enable_checkpointing=True,
        enable_model_summary=True,
        default_root_dir=output_dir,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
    )

    # 4) train
    trainer.fit(model, train_loader, val_loader)

    # 5) test with the best checkpoint
    best_model = PlmEssentialGeneClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(best_model, test_loader)

    # 6) save per‑protein predictions
    test_df["prob"] = best_model.test_probs
    # compute auroc and auprc
    auroc_test = roc_auc_score(test_df["label"], test_df["prob"])
    auprc_test = average_precision_score(test_df["label"], test_df["prob"])
    print(f"Test AUROC: {auroc_test:.4f}, AUPRC: {auprc_test:.4f}")
    test_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)


class ArgumentParser(Tap):
    """Argument parser for finetuning linear model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    model_path: str = "esmc_300m"
    output_dir: str = "/tmp/"
    hidden_size: int = 640
    lr: float = 1e-5
    dropout: float = 0.2
    batch_size: int = 32
    max_seq_len: int = 1024
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    promoter_len: int = 128
    prot_dataset_path: str = "macwiatrak/bacbench-essential-genes-protein-sequences"
    dna_dataset_path: str = "macwiatrak/bacbench-essential-genes-dna"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        model_path=args.model_path,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        lr=args.lr,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        promoter_len=args.promoter_len,
        prot_dataset_path=args.prot_dataset_path,
        dna_dataset_path=args.dna_dataset_path,
    )
