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
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from bacbench.modeling.embed_dna import chunk_dna_sequence, get_dna_seq


def grouped_mean_pytorch(M: torch.Tensor, K: torch.Tensor):
    """
    Average rows in `M` that share the same key in `K`.

    Parameters
    ----------
    M : (B, D) tensor  – any floating dtype (f16, bf16, f32, …)
    K : (B,)   int tensor

    Returns
    -------
    means  : (L, D) tensor  – mean per unique key, same dtype as M
    groups : (L,)   tensor  – the unique keys (sorted as returned by torch.unique)
    """
    # ensure keys live on the same device as M
    K = K.to(M.device)

    # map each row to a contiguous group index
    groups, inverse = torch.unique(K, sorted=True, return_inverse=True)  # (L,), (B,)

    # allocate accumulator in *exactly the same dtype* as M
    sums = torch.zeros(groups.numel(), M.size(1), device=M.device, dtype=M.dtype)  # <- dtype fix
    sums.index_add_(0, inverse, M)  # accumulate

    # counts per group, then divide (cast counts to dtype of sums)
    counts = torch.bincount(inverse, minlength=groups.numel()).unsqueeze(1)
    means = sums / counts.to(sums.dtype)

    return means


# ------------------------- data helpers --------------------------------
class DNADataset(Dataset):
    """Wrap a pandas DataFrame with 'sequence' and 'label' columns."""

    def __init__(self, df: pd.DataFrame, seqs: dict, promoter_len: int, max_seq_len: int, overlap: int):
        self.df = df.reset_index(drop=True)
        self.seqs = seqs
        self.promoter_len = promoter_len
        self.max_seq_len = max_seq_len
        self.overlap = overlap

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = get_dna_seq(
            dna_seq=self.seqs[row.genome_name],
            start=int(row.start),
            end=int(row.end),
            strand=row.strand,
            promoter_len=self.promoter_len,
        )
        seqs = chunk_dna_sequence(
            seq,
            max_seq_len=self.max_seq_len,
            overlap=self.overlap,
        )
        return {"seqs": seqs, "label": row.label}


def collate_dna_seqs(tokenizer, max_seq_len, batch):
    """Collate function for DNA sequences."""
    seqs = []
    seq_indices = []
    for idx, b in enumerate(batch):
        seqs += b["seqs"]
        seq_indices += [idx] * len(b["seqs"])
    inputs = tokenizer.batch_encode_plus(
        seqs, return_tensors="pt", padding="longest", truncation=True, max_length=max_seq_len
    )
    inputs["seq_indices"] = torch.tensor(seq_indices, dtype=torch.long)
    inputs["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return inputs


def load_model(model_path: str):
    """Load a pretrained DNA language model and its tokenizer."""
    if "nucleotide-transformer" in model_path:
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer, "nucleotide_transformer"

    if "dnabert" in model_path:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer, "dnabert"

    if "Mistral-DNA" in model_path:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer, "mistral"

    raise ValueError(f"Unsupported model type: {model_path}. Supported: nucleotide_transformer, dnabert and mistral.")


# ------------------------- Lightning module ----------------------------
class DNALMEssentialGeneClassifier(pl.LightningModule):
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
        if self.model_type == "nucleotide_transformer":
            last_hidden_state = self.model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                encoder_attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )["hidden_states"][-1]
        elif self.model_type == "dnabert":
            last_hidden_state = self.model(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
            )[0]
        elif self.model_type == "mistral":
            last_hidden_state = self.model(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
            ).last_hidden_state
        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. Supported: nucleotide_transformer, dnabert, mistral."
            )
        out = torch.einsum(
            "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
        ) / inputs["attention_mask"].sum(1).unsqueeze(1)
        out = grouped_mean_pytorch(out, inputs["seq_indices"])
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
    overlap: int = 32,
    dataset_path: str = "macwiatrak/bacbench-essential-genes-protein-sequences",
):
    """Finetune a pretrained protein language model on essential gene classification."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset(dataset_path)  # HF streaming

    train_df = ds["train"].to_pandas()
    train_seqs = {row.genome_name: row.dna_seq for _, row in train_df.iterrows()}
    train_df = train_df.drop(columns=["dna_seq"]).explode(
        ["protein_id", "product", "start", "end", "strand", "essential"]
    )
    train_df["label"] = train_df.essential.map({"Yes": 1, "No": 0})

    val_df = ds["validation"].to_pandas()
    val_seqs = {row.genome_name: row.dna_seq for _, row in val_df.iterrows()}
    val_df = val_df.drop(columns=["dna_seq"]).explode(["protein_id", "product", "start", "end", "strand", "essential"])
    val_df["label"] = val_df.essential.map({"Yes": 1, "No": 0})

    test_df = ds["test"].to_pandas()
    test_seqs = {row.genome_name: row.dna_seq for _, row in test_df.iterrows()}
    test_df = test_df.drop(columns=["dna_seq"]).explode(
        ["protein_id", "product", "start", "end", "strand", "essential"]
    )
    test_df["label"] = test_df.essential.map({"Yes": 1, "No": 0})

    model, tokenizer, model_type = load_model(model_path)
    for p in model.parameters():
        p.requires_grad = True

    # 2) datasets & dataloaders
    train_ds = DNADataset(train_df, train_seqs, promoter_len, max_seq_len, overlap)
    val_ds = DNADataset(val_df, val_seqs, promoter_len, max_seq_len, overlap)
    test_ds = DNADataset(test_df, test_seqs, promoter_len, max_seq_len, overlap)

    collate_fn = partial(collate_dna_seqs, tokenizer, max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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
    model = DNALMEssentialGeneClassifier(
        model=model, hidden_size=hidden_size, lr=lr, dropout=dropout, model_type=model_type
    )

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
    best_model = DNALMEssentialGeneClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
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

    model_path: str = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
    output_dir: str = "/tmp/"
    hidden_size: int = 768
    lr: float = 1e-5
    dropout: float = 0.2
    batch_size: int = 32
    max_seq_len: int = 2048
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    promoter_len: int = 128
    overlap: int = 32
    dataset_path: str = "macwiatrak/bacbench-essential-genes-dna"


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
        overlap=args.overlap,
        dataset_path=args.dataset_path,
    )
