from __future__ import annotations

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from tap import Tap
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# apply user-level warning filters after imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# learnigng rates for different models after tuning on the validation set
MODEL2LR = {
    "gLM2": 0.001,
    "ProkBERT": 0.01,
    "esm2": 0.01,
    "bacformer": 0.01,
    "bacformer_wo_pretraining": 0.01,
    "dnabert": 0.001,
    "esmc": 0.001,
    "esmc_large": 0.001,
    "mistral_dna": 0.005,
    "nucleotide_transformer": 0.005,
    "protbert": 0.005,
    "evo": 0.0005,
}


# ------------------------- utilities -------------------------
def _to_numpy_matrix(series) -> np.ndarray:
    """Stack a column of lists/arrays into a 2D numpy matrix.

    Parameters
    ----------
    series : pd.Series
        Series where each element is a list or array of features.

    Returns
    -------
    np.ndarray
        2D array of shape (n_samples, n_features).
    """
    arrs = series.apply(lambda x: np.asarray(x, dtype=np.float32)).to_list()
    X = np.stack(arrs, axis=0)
    return X


def _encode_labels(y: pd.Series) -> tuple[np.ndarray, dict]:
    """Factorize labels -> {class: idx} and return y_int, mapping.

    Parameters
    ----------
    y : pd.Series
        Series of labels (strings or other hashable types).

    Returns
    -------
    tuple[np.ndarray, dict]
        y_int: Integer-encoded labels as a numpy array.
        cls2id: Dictionary mapping class names to integer IDs.
    """
    classes = sorted(pd.Series(y.unique()).astype(str).tolist())
    cls2id = {c: i for i, c in enumerate(classes)}
    # map via string, to be robust to mixed dtypes; ensure plain numpy ndarray
    y_int = y.astype(str).map(cls2id).to_numpy(dtype=np.int64)
    return y_int, cls2id


def _filter_min_per_class(df: pd.DataFrame, label_col: str, min_per_class: int) -> pd.DataFrame:
    """Drop rows of classes with < min_per_class.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_col : str
        Column name containing class labels.
    min_per_class : int
        Minimum number of samples required per class.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only classes with sufficient samples.
    """
    counts = df[label_col].value_counts(dropna=True)
    keep = counts[counts >= min_per_class].index
    return df[df[label_col].isin(keep)].copy()


def _split_indices(
    y: np.ndarray,
    groups: np.ndarray | None,
    split_mode: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test indices according to split_mode.

    Parameters
    ----------
    y : np.ndarray
        Target array.
    groups : np.ndarray | None
        Group labels for group-based splitting.
    split_mode : str
        Name of the split column or mode.
    train_size : float
        Proportion of data for training.
    val_size : float
        Proportion of data for validation.
    test_size : float
        Proportion of data for testing.
    seed : int
        Random seed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Indices for training, validation, and test sets.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    n = len(y)
    all_idx = np.arange(n)

    if split_mode is not None and groups is not None:
        # group-wise split to avoid leakage across groups
        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        tr_idx, tmp_idx = next(gss1.split(all_idx, y, groups=groups))

        tmp_groups = groups[tmp_idx]
        val_prop = val_size / (val_size + test_size)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=val_prop, random_state=seed + 1)
        va_rel, te_rel = next(gss2.split(tmp_idx, y[tmp_idx], groups=tmp_groups))
        va_idx, te_idx = tmp_idx[va_rel], tmp_idx[te_rel]

    else:
        # stratified random split by labels
        sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        tr_idx, tmp_idx = next(sss1.split(all_idx, y))

        val_prop = val_size / (val_size + test_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_prop, random_state=seed + 1)
        va_rel, te_rel = next(sss2.split(tmp_idx, y[tmp_idx]))
        va_idx, te_idx = tmp_idx[va_rel], tmp_idx[te_rel]

    return tr_idx, va_idx, te_idx


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Create a dense one-hot matrix for the provided integer labels."""
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _macro_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Compute robust macro metrics given integer labels and class probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        True integer labels.
    proba : np.ndarray
        Predicted class probabilities of shape (n_samples, n_classes).

    Returns
    -------
    dict[str, float]
        Dictionary containing macro_auroc, macro_auprc, macro_f1, macro_accuracy, and accuracy.
    """
    num_classes = proba.shape[1]
    y_pred = proba.argmax(axis=1)

    # Macro AUROC
    try:
        if num_classes == 2:
            auroc = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            present = np.unique(y_true)
            # restrict to present classes
            auroc = float(roc_auc_score(y_true, proba[:, present], multi_class="ovr", average="macro", labels=present))
    except Exception:  # noqa
        auroc = float("nan")

    # Macro AUPRC
    try:
        if num_classes == 2:
            auprc = float(average_precision_score(y_true, proba[:, 1]))
        else:
            present = np.unique(y_true)
            y_oh = _one_hot(y_true, num_classes)[:, present]
            pr_per_class = average_precision_score(y_oh, proba[:, present], average=None)
            auprc = float(np.nanmean(pr_per_class))
    except Exception:  # noqa
        auprc = float("nan")

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    macro_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))

    return {
        "macro_auroc": auroc,
        "macro_auprc": auprc,
        "macro_f1": macro_f1,
        "macro_accuracy": macro_acc,
        "accuracy": acc,
    }


# ------------------------- Lightning module -------------------------


class LinearHead(pl.LightningModule):
    """LayerNorm + Dropout + Linear classifier with CE loss, logs macro metrics."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float = 1e-2,
        dropout: float = 0.1,
        class_weight: list[float] | None = None,
    ):
        """Initialize the LinearHead model.

        Parameters
        ----------
        input_dim : int
            Dimension of input features.
        output_dim : int
            Number of output classes.
        lr : float, optional
            Learning rate, by default 1e-2.
        dropout : float, optional
            Dropout probability, by default 0.1.
        class_weight : list[float] | None, optional
            Weight for each class in CE loss (for imbalanced classification), by default None.
        """
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )
        self.lr = lr

        if class_weight is not None:
            w = torch.tensor(class_weight, dtype=torch.float32)
            self.register_buffer("ce_weight", w)
        else:
            self.ce_weight = None  # not a buffer, just the attribute

        # buffers for epoch metrics
        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

        self._test_logits: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, output_dim).
        """
        return self.net(x)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy loss optionally using class weights."""
        return F.cross_entropy(logits, y, weight=self.ce_weight)

    def training_step(self, batch, batch_idx):
        """Training step for the linear head.

        Parameters
        ----------
        batch : tuple
            Tuple of (x, y) tensors.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the linear head.

        Parameters
        ----------
        batch : tuple
            Tuple of (x, y) tensors.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self._val_logits.append(logits.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_end(self):
        """Compute and log macro metrics at the end of validation epoch."""
        if self._val_logits:
            logits = torch.cat(self._val_logits, dim=0).numpy()
            targets = torch.cat(self._val_targets, dim=0).numpy()
            proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            metrics = _macro_metrics(targets, proba)
            self.log("val_macro_auroc", metrics["macro_auroc"], prog_bar=True, on_epoch=True)
            self.log("val_macro_auprc", metrics["macro_auprc"], on_epoch=True)
            self.log("val_macro_f1", metrics["macro_f1"], on_epoch=True)
            self.log("val_macro_accuracy", metrics["macro_accuracy"], on_epoch=True)
            # clear buffers
            self._val_logits.clear()
            self._val_targets.clear()

    def test_step(self, batch, batch_idx):
        """Test step for the linear head.

        Parameters
        ----------
        batch : tuple
            Tuple of (x, y) tensors.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self._test_logits.append(logits.detach().cpu())
        self._test_targets.append(y.detach().cpu())
        self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def on_test_epoch_end(self):
        """Compute and log macro metrics at the end of test epoch."""
        if self._test_logits:
            logits = torch.cat(self._test_logits, dim=0).numpy()
            targets = torch.cat(self._test_targets, dim=0).numpy()
            proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            metrics = _macro_metrics(targets, proba)
            # log for completeness
            for k, v in metrics.items():
                self.log(f"test_{k}", v, on_epoch=True)
            self.test_results_ = metrics  # stash for retrieval
            self._test_logits.clear()
            self._test_targets.clear()

    def configure_optimizers(self):
        """Configure the optimizer for the linear head.

        Returns
        -------
        torch.optim.Optimizer
            Configured AdamW optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ------------------------- main helpers -------------------------


def filter_phenotypes(
    df: pd.DataFrame,
    phenotype_cols: list[str],
    min_class_samples: int = 50,
):
    """Drop phenotype columns that don't have >=2 classes each with at least min_class_samples.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing phenotype columns.
    phenotype_cols : list[str]
        List of phenotype column names to check.
    min_class_samples : int, optional
        Minimum samples per class required, by default 50.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only valid phenotype columns.
    """
    keep_cols = []
    for col in phenotype_cols:
        vc = df[col].dropna().value_counts()
        vc = {v: c for v, c in vc.items() if c >= min_class_samples}
        if len(vc) >= 2:
            keep_cols.append(col)
    # Keep all metadata columns + filtered phenotypes
    base_cols = df.columns[:5] if len(df.columns) >= 16 else df.columns
    kept = list(base_cols) + keep_cols
    kept = [c for c in kept if c in df.columns]
    return df[kept].copy()


def _make_loaders(
    X_train, y_train, X_val, y_val, X_test=None, y_test=None, batch_size: int = 256, num_workers: int = 4
):
    """Build Torch DataLoaders for train/val/test splits."""
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    if X_test is not None and y_test is not None:
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_dl = None
    return train_dl, val_dl, test_dl


def train_and_predict(
    df: pd.DataFrame,
    model_name: str,
    lr: float,
    phenotype: str,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    split: str = "genus",
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    test_after_train: bool = False,
    seed: int = 1,
):
    """Train and predict phenotypic traits using a linear model with LN+Dropout.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    model_name : str
        Name of the model (column name for features).
    lr : float
        Learning rate.
    phenotype : str
        Name of the phenotype (target column).
    max_epochs : int, optional
        Maximum training epochs, by default 100.
    early_stopping_patience : int, optional
        Patience for early stopping, by default 10.
    split : str, optional
        Split mode or column name, by default "genus".
    train_size : float, optional
        Training set proportion, by default 0.7.
    val_size : float, optional
        Validation set proportion, by default 0.1.
    test_size : float, optional
        Test set proportion, by default 0.2.
    test_after_train : bool, optional
        Whether to run evaluation on test set after training, by default False.
    seed : int, optional
        Random seed, by default 1.

    Returns
    -------
    dict
        Dictionary containing results and metrics.
    """
    pl.seed_everything(seed, workers=True)

    # Select columns and drop NaNs
    cols = [model_name, split, phenotype]
    sub = df[cols].dropna().reset_index(drop=True)
    if len(sub) == 0:
        return {"phenotype": phenotype, "seed": seed, "model_name": model_name, "skipped": "no_data"}

    # Features and labels
    X = _to_numpy_matrix(sub[model_name])
    y_int, cls2id = _encode_labels(sub[phenotype])
    num_classes = len(cls2id)
    input_dim = X.shape[1]
    output_dim = num_classes

    # Split
    groups = sub[split].astype(str).to_numpy() if split is not None else None
    tr_idx, va_idx, te_idx = _split_indices(
        y=y_int,
        groups=groups,
        split_mode=split,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    Xtr, ytr = X[tr_idx], y_int[tr_idx]
    Xva, yva = X[va_idx], y_int[va_idx]
    Xte, yte = X[te_idx], y_int[te_idx]

    train_dl, val_dl, test_dl = _make_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=256, num_workers=4)

    # Model
    lit_model = LinearHead(input_dim=input_dim, output_dim=output_dim, lr=lr, dropout=0.1, class_weight=None)

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        monitor="val_macro_auroc",
        mode="max",
        save_top_k=1,
        save_last=False,
        filename=f"{phenotype}-{{epoch:02d}}-{{val_macro_auroc:.4f}}",
    )
    es_cb = EarlyStopping(monitor="val_macro_auroc", mode="max", patience=early_stopping_patience, min_delta=0.0)

    # Logger (CSV)
    # logger = CSVLogger(save_dir=os.getcwd(), name=f"logs_{model_name}_{phenotype}", flush_logs_every_n_steps=50)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        deterministic=False,
        callbacks=[ckpt_cb, es_cb],
        # logger=logger,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(lit_model, train_dl, val_dl)

    # Load best
    best_path = ckpt_cb.best_model_path
    if best_path and os.path.exists(best_path):
        lit_model = LinearHead.load_from_checkpoint(best_path)

    # Determine inference device and ensure best model parameters are on it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()
    with torch.no_grad():
        # Evaluate on the validation split using the best checkpoint
        logits_val = []
        y_val_all = []
        for xb, yb in val_dl:
            logits_val.append(lit_model(xb.to(device)).cpu())
            y_val_all.append(yb.cpu())
        logits_val = torch.cat(logits_val, 0).numpy()
        y_val_all = torch.cat(y_val_all, 0).numpy()
        proba_val = torch.softmax(torch.from_numpy(logits_val), dim=1).numpy()
        val_metrics = _macro_metrics(y_val_all, proba_val)

    result = {
        "phenotype": phenotype,
        "seed": seed,
        "model_name": model_name,
        "split": split,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "val_macro_auroc": val_metrics["macro_auroc"],
        "val_macro_auprc": val_metrics["macro_auprc"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_macro_accuracy": val_metrics["macro_accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "best_ckpt": best_path or "",
    }

    if test_after_train and test_dl is not None:
        # Lightning test to reuse code paths (will stash test_results_)
        trainer.test(lit_model, test_dl, verbose=False)
        test_metrics = getattr(lit_model, "test_results_", None)
        if test_metrics is None:
            # Fallback manual: handle cases where Lightning didn't cache test metrics
            logits_test = []
            y_test_all = []
            for xb, yb in test_dl:
                logits_test.append(lit_model(xb.to(device)).cpu())
                y_test_all.append(yb.cpu())
            logits_test = torch.cat(logits_test, 0).numpy()
            y_test_all = torch.cat(y_test_all, 0).numpy()
            proba_test = torch.softmax(torch.from_numpy(logits_test), dim=1).numpy()
            test_metrics = _macro_metrics(y_test_all, proba_test)

        result.update(
            {
                "test_macro_auroc": test_metrics["macro_auroc"],
                "test_macro_auprc": test_metrics["macro_auprc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_macro_accuracy": test_metrics["macro_accuracy"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

    return result


def run(
    df: pd.DataFrame,
    model_name: str,
    lr: float | None = None,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    min_class_samples: int = 50,
    split: str = "genus",
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    test_after_train: bool = False,
    seeds: list[int] | None = None,
    limit_n_phenotypes: int | None = None,
):
    """Run the training and prediction for phenotypic traits.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    model_name : str
        Name of the model (feature column).
    lr : float | None, optional
        Learning rate. If None, uses default from MODEL2LR. By default None.
    max_epochs : int, optional
        Maximum training epochs, by default 100.
    early_stopping_patience : int, optional
        Patience for early stopping, by default 10.
    min_class_samples : int, optional
        Minimum samples per class required, by default 50.
    split : str, optional
        Split mode or column name, by default "genus".
    train_size : float, optional
        Training set proportion, by default 0.7.
    val_size : float, optional
        Validation set proportion, by default 0.1.
    test_size : float, optional
        Test set proportion, by default 0.2.
    test_after_train : bool, optional
        Whether to run evaluation on test set, by default False.
    seeds : list[int] | None, optional
        List of random seeds to run. If None, uses [1]. By default None.
    limit_n_phenotypes : int | None, optional
        Limit number of phenotypes to process (for debugging), by default None.

    Returns
    -------
    pd.DataFrame
        Dataframe containing aggregated results for all phenotypes and seeds.
    """
    # Identify phenotype columns and filter them globally
    if seeds is None:
        seeds = [1]
    all_pheno_cols = list(df.columns[5:])  # as specified
    filtered_df = filter_phenotypes(df, all_pheno_cols, min_class_samples)
    phenotype_cols = list(filtered_df.columns[5:])
    if limit_n_phenotypes is not None:
        phenotype_cols = phenotype_cols[:limit_n_phenotypes]

    if lr is None:
        lr = MODEL2LR[model_name]

    out = []
    for seed in seeds:
        for phenotype in tqdm(phenotype_cols):
            res = train_and_predict(
                filtered_df,
                model_name=model_name,
                lr=lr,
                phenotype=phenotype,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                split=split,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                test_after_train=test_after_train,
                seed=seed,
            )
            out.append(res)

    out_df = pd.DataFrame(out)

    # Report mean metrics across phenotypes and seeds (validation metrics)
    metric_cols = ["val_macro_auroc", "val_macro_auprc", "val_macro_f1", "val_macro_accuracy", "val_accuracy"]
    available = [m for m in metric_cols if m in out_df.columns]
    if available:
        means = out_df[available].mean(numeric_only=True)
        print("\n=== Mean validation metrics across phenotypes/seeds ===")
        for k, v in means.items():
            print(f"{k}: {v:.4f}")

    # Optional: similar report for test if present
    test_metric_cols = ["test_macro_auroc", "test_macro_auprc", "test_macro_f1", "test_macro_accuracy", "test_accuracy"]
    available_test = [m for m in test_metric_cols if m in out_df.columns]
    if test_after_train and available_test:
        means_test = out_df[available_test].mean(numeric_only=True)
        print("\n=== Mean test metrics across phenotypes/seeds ===")
        for k, v in means_test.items():
            print(f"{k}: {v:.4f}")

    return out_df


# ------------------------- CLI -------------------------


class ArgParser(Tap):
    """Arguments for phenotype linear probing."""

    def __init__(self):
        """Initialize arguments."""
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_genomes_df_filepath: str
    labels_df_filepath: str
    output_dir: str
    model_name: str
    lr: float
    max_epochs: int = 100
    early_stopping_patience: int = 10
    min_class_samples: int = 50
    split: str = "genus"
    train_size: float = 0.7
    val_size: float = 0.1
    test_size: float = 0.2
    test_after_train: bool = True
    limit_n_phenotypes: int | None = None  # limit number of phenotypes to process, for debugging


if __name__ == "__main__":
    args = ArgParser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.input_genomes_df_filepath)
    labels_df = pd.read_parquet(args.labels_df_filepath)
    n_before_merge = len(df)
    df = df.merge(labels_df, on="genome_id", how="inner")
    n_after_merge = len(df)
    assert n_before_merge == n_after_merge, "Merging with labels changed number of genomes! Investigate it."

    today = datetime.today().strftime("%Y_%m_%d")
    print(f"\nRunning phenotype prediction for model: {args.model_name}")
    metrics_df = run(
        df=df,
        model_name=args.model_name,
        lr=args.lr,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_class_samples=args.min_class_samples,
        split=args.split,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        test_after_train=args.test_after_train,
        seeds=[1, 2, 3],
        limit_n_phenotypes=args.limit_n_phenotypes,
    )
    out_path = os.path.join(args.output_dir, f"phenotypic_traits_preds_{args.model_name}_{today}.csv")
    metrics_df.to_csv(out_path, index=False)
    print(f"\nSaved metrics to: {out_path}")
