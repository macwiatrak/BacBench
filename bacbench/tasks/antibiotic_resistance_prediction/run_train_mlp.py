"""AMR k‑fold evaluation pipeline

================================
Implements the protocol you requested:
1. Stratified *k*‑fold split → (train + val) | test.
2. 20 % of the training part becomes a **stratified validation set**.
3. Train on the remaining 80 %, early‑stop on **validation AUPRC** (default, configurable).
4. After training, sweep thresholds on the validation set to find the one that maximises F1.
5. Evaluate the locked model + threshold on the fold’s test set and report metrics.
"""

# -----------------------------------------------------------------------------
import io
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import tqdm
from lightning_fabric.utilities import cloud_io  # <-- underscore here!
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tap import Tap
from torch import nn
from torch.nn.functional import softplus
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import AUROC, AveragePrecision, F1Score
from torchmetrics.functional import pearson_corrcoef, r2_score
from torchmetrics.regression import PearsonCorrCoef, R2Score


# hot fix for Lightning/Fabric
def _atomic_save_same_fs(checkpoint, filepath):
    """Write `filepath` atomically inside its own directory so we never cross filesystems."""
    # serialize to an in-memory buffer
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)

    # tmp file lives in target dir ⇒ same FS
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(filepath))
    with os.fdopen(fd, "wb") as tmp_file:
        tmp_file.write(buffer.getvalue())

    shutil.move(tmp_path, filepath)  # now rename is intra-FS → always works


# monkey-patch Lightning/Fabric
cloud_io._atomic_save = _atomic_save_same_fs


# -----------------------------------------------------------------------------
#  MODEL
# -----------------------------------------------------------------------------
class BinaryMLP(pl.LightningModule):
    """One‑hidden‑layer MLP for binary classification."""

    def __init__(
        self, input_dim: int, dropout: float = 0.2, hidden_dim: int = 128, lr: float = 0.001, regression: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.lr = lr
        self.regression = regression

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if regression:
            self.criterion = nn.MSELoss()
            self.val_pearson = PearsonCorrCoef()
            self.val_r2 = R2Score()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            # threshold‑free metrics for monitoring
            self.val_auroc = AUROC(task="binary")
            self.val_auprc = AveragePrecision(task="binary")
            # F1 @ 0.5 just for reference
            self.val_f1 = F1Score(task="binary", threshold=0.5)

    def forward(self, x):
        """Forward method"""
        return self.model(x).squeeze(1)

    def _shared_step(self, batch):
        """Shared step for training and validation."""
        x, y = batch
        logits = self.forward(x)

        # if regression pass it through softplus
        if self.regression:
            logits = softplus(logits)

        loss = self.criterion(logits, y.float())

        if not self.regression:
            logits = torch.sigmoid(logits)
        return loss, logits, y

    def training_step(self, batch, _):
        """Training step."""
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        """Validation step."""
        loss, probs, y = self._shared_step(batch)
        if self.regression:
            self.val_pearson.update(probs, y)
            self.val_r2.update(probs, y)
        else:
            self.val_auroc.update(probs, y)
            self.val_auprc.update(probs, y)
            self.val_f1.update(probs, y)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """Log validation metrics at the end of the epoch."""
        if self.regression:
            self.log_dict(
                {
                    "val_pearson": self.val_pearson.compute(),
                    "val_r2": self.val_r2.compute(),
                },
                prog_bar=True,
            )
            for m in (self.val_pearson, self.val_r2):
                m.reset()
        else:
            self.log_dict(
                {
                    "val_auroc": self.val_auroc.compute(),
                    "val_auprc": self.val_auprc.compute(),
                    "val_f1": self.val_f1.compute(),
                },
                prog_bar=True,
            )
            for m in (self.val_auroc, self.val_auprc, self.val_f1):
                m.reset()

    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# -----------------------------------------------------------------------------
#  UTILITIES
# -----------------------------------------------------------------------------


def _seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _best_threshold(probs: np.ndarray, labels: np.ndarray, resolution: int = 100) -> float:
    """Return the threshold in [0,1] (inclusive) that maximises F1 on *labels*."""
    ts = np.linspace(0.0, 1.0, resolution + 1)
    f1s = [f1_score(labels, probs >= t, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f1s))])


# -----------------------------------------------------------------------------
#  K‑FOLD TRAIN / VAL / TEST
# -----------------------------------------------------------------------------


def train_kfold_mlp(
    X: np.ndarray,
    y: np.ndarray,
    dim: int,
    k: int,
    *,
    dropout: float = 0.2,
    lr: float = 0.001,
    regression: bool = False,
    early_stop_patience: int = 10,
    epochs: int = 50,
    output_dir: str = "./output",
    seed: int = 42,
    batch_size: int = 128,
    monitor_metric: str = "val_auprc",
    val_frac: float = 0.2,
    num_workers: int = 7,
) -> pd.DataFrame:
    """Per‑fold training and evaluation following the requested protocol."""
    assert X.shape[1] == dim
    os.makedirs(output_dir, exist_ok=True)
    _seed_everything(seed)

    fold_results: list[dict] = []

    if regression:
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    else:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (idx_train_full, idx_test) in enumerate(splitter.split(X, y), 1):
        # ------------------------------------------------------------------
        #  Split indices
        # ------------------------------------------------------------------
        idx_train, idx_val = train_test_split(
            idx_train_full,
            test_size=val_frac,
            stratify=None if regression else y[idx_train_full],  # <-- NEW
            random_state=seed,
        )

        def _make_loader(idxs, shuffle=False, num_workers=4):
            label_dtype = torch.float32 if regression else torch.long
            ds = TensorDataset(
                torch.tensor(X[idxs], dtype=torch.float32),
                torch.tensor(y[idxs], dtype=label_dtype),
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        train_loader = _make_loader(idx_train, shuffle=True, num_workers=num_workers)
        val_loader = _make_loader(idx_val, num_workers=num_workers)
        test_loader = _make_loader(idx_test, num_workers=num_workers)

        # ------------------------------------------------------------------
        #  Train model
        # ------------------------------------------------------------------
        model = BinaryMLP(dim, dropout=dropout, lr=lr, regression=regression)
        ckpt_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_cb = ModelCheckpoint(
            monitor=monitor_metric,
            mode="max",
            dirpath=ckpt_dir,
            filename="best",
            save_top_k=1,
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            deterministic=True,
            callbacks=[
                EarlyStopping(monitor=monitor_metric, mode="max", patience=early_stop_patience),
                checkpoint_cb,
            ],
            enable_model_summary=False,
            logger=False,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_loader, val_loader)

        best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
        best_model = BinaryMLP.load_from_checkpoint(best_path) if best_path else model
        best_model.eval()
        device = best_model.device

        # ------------------------------------------------------------------
        #  Choose threshold on validation set
        # ------------------------------------------------------------------
        if not regression:
            with torch.no_grad():
                val_probs = torch.cat([torch.sigmoid(best_model(xb.to(device))) for xb, _ in val_loader]).cpu().numpy()
                val_labels = np.concatenate([yb.numpy() for _, yb in val_loader])
            thr = _best_threshold(val_probs, val_labels)

        # ------------------------------------------------------------------
        #  Evaluate on test set with chosen threshold
        # ------------------------------------------------------------------
        with torch.no_grad():
            if regression:
                test_preds = torch.cat([softplus(best_model(xb.to(device))) for xb, _ in test_loader]).cpu()
            else:
                test_probs = (
                    torch.cat([torch.sigmoid(best_model(xb.to(device))) for xb, _ in test_loader]).cpu().numpy()
                )
                test_preds = (test_probs >= thr).astype(int)
            test_labels = np.concatenate([yb.numpy() for _, yb in test_loader])

        if regression:
            fold_results.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "test_pearson": pearson_corrcoef(test_preds, torch.tensor(test_labels)).item(),
                    "test_r2": r2_score(test_preds, torch.tensor(test_labels)).item(),
                }
            )
        else:
            fold_results.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "threshold": thr,
                    "test_f1": f1_score(test_labels, test_preds, zero_division=0),
                    "test_auroc": roc_auc_score(test_labels, test_probs),
                    "test_auprc": average_precision_score(test_labels, test_probs),
                }
            )
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    return pd.DataFrame(fold_results)


# -----------------------------------------------------------------------------
#  ANTIBIOTIC‑LEVEL WRAPPER
# -----------------------------------------------------------------------------


def train_eval_antibiotic(
    df: pd.DataFrame,
    model_name: str,
    antibiotic: str,
    k: int,
    *,
    output_dir: str = "/tmp/output",
    dropout: float = 0.2,
    lr: float = 0.005,
    regression: bool = False,
    early_stop_patience: int = 5,
    epochs: int = 100,
    seed: int = 42,
    batch_size: int = 128,
    monitor_metric: str = "val_auprc",
    num_workers: int = 0,
) -> pd.DataFrame:
    """Run the full k‑fold protocol for a given antibiotic label column."""
    df = df[df[antibiotic].notna()]
    X = np.stack(df[model_name].tolist()).astype(np.float32)
    y = df[antibiotic].values if regression else df[antibiotic].astype(int).values

    out = os.path.join(output_dir, antibiotic)
    os.makedirs(out, exist_ok=True)

    return train_kfold_mlp(
        X,
        y,
        dim=X.shape[1],
        k=k,
        dropout=dropout,
        lr=lr,
        regression=regression,
        early_stop_patience=early_stop_patience,
        epochs=epochs,
        output_dir=out,
        seed=seed,
        batch_size=batch_size,
        monitor_metric=monitor_metric,
        num_workers=num_workers,
    )


# -----------------------------------------------------------------------------
#  EXAMPLE USAGE (uncomment to run)
# -----------------------------------------------------------------------------


class ArgumentParser(Tap):
    """Argument parser for training Bacformer (Lightning version)."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_filepath: str
    output_dir: str
    k_folds: int = 5
    lr: float = 0.005
    regression: bool = False
    dropout: float = 0.2
    early_stop_patience: int = 5
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 0
    model_name: str = "bacformer"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    df = pd.read_parquet(args.input_filepath)

    assert df.columns[:2].tolist() == ["genome_name", args.model_name], (
        "First two columns must be genome_name and model_name"
    )
    antibiotics = df.columns[2:]
    output = []

    monitor_metric = "val_r2" if args.regression else "val_auprc"
    for ant in tqdm(antibiotics):
        print(f"\nRunning {args.model_name} on {ant}...\n")
        for seed in [1, 2, 3]:
            metrics = train_eval_antibiotic(
                df,
                model_name=args.model_name,
                antibiotic=ant,
                k=args.k_folds,
                output_dir=args.output_dir,
                epochs=args.epochs,
                dropout=args.dropout,
                early_stop_patience=args.early_stop_patience,
                seed=seed,
                batch_size=args.batch_size,
                monitor_metric=monitor_metric,
                num_workers=args.num_workers,
                lr=args.lr,
                regression=args.regression,
            )
            metrics["antibiotic"] = ant
            metrics["method"] = args.model_name
            output.append(metrics)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_df = pd.concat(output)
        output_df.to_csv(
            os.path.join(args.output_dir, f"results_{args.model_name}.csv"),
            index=False,
        )
