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
from tap import Tap
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# apply user-level warning filters after imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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

    For binary classification, ensures the positive class is always mapped to 1
    and the negative class to 0. This is important for metrics like AUROC and
    average precision that assume the positive class has label 1.

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
    # Get unique classes as strings for consistent handling
    classes = sorted(pd.Series(y.unique()).astype(str).tolist())
    num_classes = len(classes)

    if num_classes == 2:
        # Binary classification: ensure positive class maps to 1, negative to 0
        # Define known negative labels (case-insensitive)
        negative_labels = {"no", "0", "-", "false", "negative", "neg", "n"}
        # Define known positive labels (case-insensitive)
        positive_labels = {"yes", "1", "+", "true", "positive", "pos", "y"}

        # Identify which class is negative and which is positive
        class_lower = [c.lower() for c in classes]

        # Check if first class (alphabetically) is a known negative label
        first_is_negative = class_lower[0] in negative_labels
        first_is_positive = class_lower[0] in positive_labels
        second_is_negative = class_lower[1] in negative_labels
        second_is_positive = class_lower[1] in positive_labels

        if first_is_negative or second_is_positive:
            # First class is negative -> map to 0, second class -> 1 (default order works)
            cls2id = {classes[0]: 0, classes[1]: 1}
        elif first_is_positive or second_is_negative:
            # First class is positive -> swap: first class -> 1, second class -> 0
            cls2id = {classes[0]: 1, classes[1]: 0}
        else:
            # Unknown labels: fall back to alphabetical order (first=0, second=1)
            # This maintains backward compatibility
            cls2id = {c: i for i, c in enumerate(classes)}
    else:
        # Multi-class: use alphabetical ordering
        cls2id = {c: i for i, c in enumerate(classes)}

    # Map labels via string to be robust to mixed dtypes; ensure plain numpy ndarray
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
    split_mode: str | None,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test indices according to split_mode.

    Strategy:
      1) If groups are provided and at least three groups exist, split by group.
      2) Else stratify by label when each class can appear in train/val/test.
      3) Else fall back to a sample-level random split.

    Parameters
    ----------
    y : np.ndarray
        Target array.
    groups : np.ndarray | None
        Group labels for group-based splitting.
    split_mode : str | None
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

    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(y))

    def _split_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.array(values, copy=True)
        rng.shuffle(values)
        n = len(values)
        if n < 3:
            raise ValueError("At least three samples or groups are required for train/val/test splitting.")
        val_prop = val_size / (val_size + test_size)
        n_train = min(max(1, int(round(train_size * n))), n - 2)
        tmp = values[n_train:]
        n_val = min(max(1, int(round(val_prop * len(tmp)))), len(tmp) - 1)
        return values[:n_train], tmp[:n_val], tmp[n_val:]

    if split_mode is not None and groups is not None and len(np.unique(groups)) >= 3:
        train_groups, val_groups, test_groups = _split_values(np.unique(groups))
        tr_idx = all_idx[np.isin(groups, train_groups)]
        va_idx = all_idx[np.isin(groups, val_groups)]
        te_idx = all_idx[np.isin(groups, test_groups)]
        if len(tr_idx) > 0 and len(va_idx) > 0 and len(te_idx) > 0:
            return tr_idx, va_idx, te_idx

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) >= 2 and np.all(counts >= 3):
        train_parts = []
        val_parts = []
        test_parts = []
        for cls in classes:
            tr_cls, va_cls, te_cls = _split_values(all_idx[y == cls])
            train_parts.append(tr_cls)
            val_parts.append(va_cls)
            test_parts.append(te_cls)

        tr_idx = np.concatenate(train_parts)
        va_idx = np.concatenate(val_parts)
        te_idx = np.concatenate(test_parts)
        rng.shuffle(tr_idx)
        rng.shuffle(va_idx)
        rng.shuffle(te_idx)
        return tr_idx, va_idx, te_idx

    return _split_values(all_idx)


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

    auroc = float("nan")
    auprc = float("nan")
    if num_classes == 2:
        if np.unique(y_true).size == 2:
            auroc = float(roc_auc_score(y_true, proba[:, 1]))
            auprc = float(average_precision_score(y_true, proba[:, 1]))
    else:
        auroc_per_class = []
        auprc_per_class = []
        for cls in np.unique(y_true):
            y_binary = (y_true == cls).astype(np.int64)
            if np.unique(y_binary).size == 2:
                auroc_per_class.append(roc_auc_score(y_binary, proba[:, cls]))
                auprc_per_class.append(average_precision_score(y_binary, proba[:, cls]))
        if auroc_per_class:
            auroc = float(np.mean(auroc_per_class))
        if auprc_per_class:
            auprc = float(np.mean(auprc_per_class))

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
            # nn.LayerNorm(input_dim),
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
    base_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop phenotype columns that don't have >=2 classes each with at least min_class_samples.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing phenotype columns.
    phenotype_cols : list[str]
        List of phenotype column names to check.
    min_class_samples : int, optional
        Minimum samples per class required, by default 50.
    base_cols : list[str] | None, optional
        Non-phenotype columns to keep in the returned dataframe, by default None.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Filtered dataframe and the valid phenotype columns.
    """
    base_cols = list(dict.fromkeys(col for col in (base_cols or []) if col in df.columns))
    phenotype_cols = [col for col in dict.fromkeys(phenotype_cols) if col in df.columns and col not in base_cols]
    keep_cols = []
    for col in phenotype_cols:
        vc = df[col].dropna().value_counts()
        vc = {v: c for v, c in vc.items() if c >= min_class_samples}
        if len(vc) >= 2:
            keep_cols.append(col)
    return df[base_cols + keep_cols].copy(), keep_cols


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
    embeddings_col: str,
    lr: float,
    phenotype: str,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    split: str | None = "genus",
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    test_after_train: bool = False,
    model_name: str | None = None,
    seed: int = 1,
):
    """Train and predict phenotypic traits using a linear model with LN+Dropout.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    embeddings_col : str
        Name of the column containing the embeddings (features).
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
    model_name : str, optional
        Name of the model for saving and logging purposes, by default None.
    seed : int, optional
        Random seed, by default 1.

    Returns
    -------
    dict
        Dictionary containing results and metrics.
    """
    pl.seed_everything(seed, workers=True)

    if split is not None and split != "random" and split not in df.columns:
        warnings.warn(f"Split column {split!r} was not found; falling back to a random split.", stacklevel=2)
        split = None

    # Select columns and drop NaNs
    if split is None or split == "random":
        cols = [embeddings_col, phenotype]
    else:
        cols = [embeddings_col, split, phenotype]
    sub = df[cols].dropna().reset_index(drop=True)
    if len(sub) == 0:
        return {
            "phenotype": phenotype,
            "seed": seed,
            "embeddings_col": embeddings_col,
            "model_name": model_name,
            "split": split,
            "skipped": "no_data",
        }
    if len(sub) < 3:
        return {
            "phenotype": phenotype,
            "seed": seed,
            "embeddings_col": embeddings_col,
            "model_name": model_name,
            "split": split,
            "skipped": "not_enough_samples",
        }

    # Features and labels
    X = _to_numpy_matrix(sub[embeddings_col])
    y_int, cls2id = _encode_labels(sub[phenotype])
    num_classes = len(cls2id)
    input_dim = X.shape[1]
    output_dim = num_classes

    # Split
    groups = (
        sub[split].astype(str).to_numpy()
        if (split is not None and split != "random" and split in sub.columns)
        else None
    )
    tr_idx, va_idx, te_idx = _split_indices(
        y=y_int,
        groups=groups,
        split_mode=split if (split is not None and split != "random") else None,
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
        "embeddings_col": embeddings_col,
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
    lr: float,
    embeddings_col: str = "embeddings",
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    min_class_samples: int = 50,
    split: str | None = "genus",
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    test_after_train: bool = False,
    seeds: list[int] | None = None,
    model_name: str | None = None,
    phenotype_cols: list[str] | None = None,
    limit_n_phenotypes: int | None = None,
):
    """Run the training and prediction for phenotypic traits.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    lr : float
        Learning rate.
    embeddings_col : str, optional
        Name of the column containing the embeddings (features), by default "embeddings".
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
    model_name : str, optional
        Name of the model for saving and logging purposes, by default None.
    phenotype_cols : list[str] | None, optional
        Phenotype columns to process. If None, inferred by excluding known metadata columns.
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
    split_for_training = split
    if split_for_training is not None and split_for_training != "random" and split_for_training not in df.columns:
        warnings.warn(
            f"Split column {split_for_training!r} was not found; falling back to a random split.",
            stacklevel=2,
        )
        split_for_training = None
    if phenotype_cols is None:
        excluded = {
            "genome_name",
            embeddings_col,
            "species",
            "genus",
            "family",
            "order",
            "class",
            "phylum",
            "taxid",
            "ncbi_taxid",
        }
        if split_for_training is not None and split_for_training != "random":
            excluded.add(split_for_training)
        phenotype_cols = [col for col in df.columns if col not in excluded]
    base_cols = ["genome_name", embeddings_col]
    if split_for_training is not None and split_for_training != "random":
        base_cols.append(split_for_training)
    filtered_df, phenotype_cols = filter_phenotypes(
        df=df,
        phenotype_cols=phenotype_cols,
        min_class_samples=min_class_samples,
        base_cols=base_cols,
    )
    if limit_n_phenotypes is not None:
        phenotype_cols = phenotype_cols[:limit_n_phenotypes]

    out = []
    for seed in seeds:
        for phenotype in tqdm(phenotype_cols):
            res = train_and_predict(
                filtered_df,
                lr=lr,
                embeddings_col=embeddings_col,
                phenotype=phenotype,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                split=split_for_training,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                test_after_train=test_after_train,
                model_name=model_name,
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
    lr: float
    max_epochs: int = 100
    early_stopping_patience: int = 10
    min_class_samples: int = 50
    split: str = "genus"
    train_size: float = 0.7
    val_size: float = 0.1
    test_size: float = 0.2
    test_after_train: bool = False
    embeddings_col: str = "embeddings"  # column name in genomes df containing the features
    model_name: str = "unknown"  # name to use in outputs for this model
    limit_n_phenotypes: int | None = None  # limit number of phenotypes to process, for debugging


if __name__ == "__main__":
    args = ArgParser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.input_genomes_df_filepath, columns=["genome_name", args.embeddings_col])
    # sort by genome_name to ensure consistent order
    df = df.sort_values("genome_name").reset_index(drop=True)
    # read labels
    labels_df = pd.read_csv(args.labels_df_filepath)
    metadata_cols = {
        "genome_name",
        args.embeddings_col,
        args.split,
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "taxid",
        "ncbi_taxid",
    }
    groupings_path = os.path.join(os.path.dirname(__file__), "phenotypic_traits_grouppings.csv")
    phenotype_cols = []
    if os.path.exists(groupings_path):
        grouped_phenotypes = pd.read_csv(groupings_path)["phenotype"].tolist()
        phenotype_cols = [col for col in grouped_phenotypes if col in labels_df.columns]
    if not phenotype_cols:
        phenotype_cols = [col for col in labels_df.columns if col not in metadata_cols]
    # merge on genome_name, inner join to keep only genomes with labels (should be all if data is correct)
    n_before_merge = len(df)
    df = df.merge(labels_df, on="genome_name", how="inner")
    n_after_merge = len(df)
    assert n_before_merge == n_after_merge, "Merging with labels changed number of genomes! Investigate it."

    today = datetime.today().strftime("%Y_%m_%d")
    print(f"\nRunning phenotype prediction for model: {args.model_name}")

    metrics_df = run(
        df=df,
        embeddings_col=args.embeddings_col,
        lr=args.lr,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_class_samples=args.min_class_samples,
        split=args.split,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        test_after_train=args.test_after_train,
        model_name=args.model_name,
        phenotype_cols=phenotype_cols,
        seeds=[1, 2, 3],
        limit_n_phenotypes=args.limit_n_phenotypes,
    )

    # Report mean metrics across phenotypes and seeds (test metrics)
    print(f"Metrics for lr: {args.lr}")
    metric_cols = ["test_macro_auroc", "test_macro_auprc", "test_macro_f1", "test_macro_accuracy", "test_accuracy"]
    available = [m for m in metric_cols if m in metrics_df.columns]
    if available:
        means = metrics_df[available].mean(numeric_only=True)
        print("\n=== Mean test metrics across phenotypes/seeds ===")
        for k, v in means.items():
            print(f"{k}: {v:.4f}")

    out_path = os.path.join(args.output_dir, f"phenotypic_traits_preds_{args.model_name}_{today}.csv")
    metrics_df.to_csv(out_path, index=False)
    print(f"\nSaved metrics to: {out_path}")
