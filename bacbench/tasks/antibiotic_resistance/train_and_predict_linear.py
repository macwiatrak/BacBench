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
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit
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
    # ensure a plain numpy ndarray (not a pandas ExtensionArray)
    y_int = y.astype(str).map(cls2id).to_numpy(dtype=np.int64)
    return y_int, cls2id


def _split_indices(
    y: np.ndarray,
    groups: np.ndarray | None,
    split_mode: str | None,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    is_regression: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test indices according to split_mode, with robust fallbacks.

    Strategy:
      1) If groups provided -> GroupShuffleSplit twice.
      2) Else:
         - classification: StratifiedShuffleSplit if >=2 classes, else ShuffleSplit
         - regression: ShuffleSplit
      3) If any step raises ValueError (e.g., tiny sample set), fallback to ShuffleSplit.

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
    is_regression : bool
        Whether the task is regression (affects stratification).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Indices for training, validation, and test sets.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    n = len(y)
    all_idx = np.arange(n)
    val_prop = val_size / (val_size + test_size)

    def _fallback_shuffle():
        # Single-stage shuffle: split train vs tmp, then tmp -> val/test
        ss1 = ShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        tr_idx, tmp_idx = next(ss1.split(all_idx))
        if len(tmp_idx) == 0:
            raise ValueError("ShuffleSplit failed: tmp set is empty.")
        ss2 = ShuffleSplit(n_splits=1, train_size=val_prop, random_state=seed + 1)
        va_rel, te_rel = next(ss2.split(tmp_idx))
        return tr_idx, tmp_idx[va_rel], tmp_idx[te_rel]

    # 1) Group-aware
    if split_mode is not None and groups is not None:
        try:
            # Need at least 2 groups to train and some groups left for tmp
            gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
            tr_idx, tmp_idx = next(gss1.split(all_idx, y, groups=groups))
            tmp_groups = groups[tmp_idx]
            gss2 = GroupShuffleSplit(n_splits=1, train_size=val_prop, random_state=seed + 1)
            va_rel, te_rel = next(gss2.split(tmp_idx, y[tmp_idx], groups=tmp_groups))
            return tr_idx, tmp_idx[va_rel], tmp_idx[te_rel]
        except Exception as e:  # noqa
            # Fallback to sample-level splitting
            try:
                return _fallback_shuffle()
            except Exception as e2:  # noqa
                raise ValueError(f"Group split failed and fallback ShuffleSplit failed: {e}; {e2}") from e

    # 2) Non-group path
    try:
        if is_regression:
            return _fallback_shuffle()
        else:
            # classification
            if np.unique(y).size >= 2:
                sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
                tr_idx, tmp_idx = next(sss1.split(all_idx, y))
                sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_prop, random_state=seed + 1)
                va_rel, te_rel = next(sss2.split(tmp_idx, y[tmp_idx]))
                return tr_idx, tmp_idx[va_rel], tmp_idx[te_rel]
            else:
                # only one class -> no stratification possible
                return _fallback_shuffle()
    except Exception as e:  # noqa
        # Last-resort fallback
        try:
            return _fallback_shuffle()
        except Exception as e2:  # noqa
            raise ValueError(f"Splitting failed and fallback ShuffleSplit failed: {e}; {e2}") from e


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float] | None:
    """Binary classification metrics from positive-class scores.

    Returns None if y_true has < 2 classes (ROC/AUPRC undefined).

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_score : np.ndarray
        Predicted scores (probabilities or logits).

    Returns
    -------
    dict[str, float] | None
        Dictionary of metrics (auroc, auprc, f1, balanced_accuracy, accuracy)
        or None if fewer than 2 classes are present.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    # If only one class is present, metrics like ROC/AUPRC are undefined.
    if np.unique(y_true).size < 2:
        return None

    y_pred = (y_score >= 0.5).astype(int)

    # These may still raise if edge-case, so guard them.
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:  # noqa
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_score))
    except Exception:  # noqa
        auprc = float("nan")

    f1 = float(f1_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics: R2 and Pearson correlation.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary containing 'r2' and 'pearson' correlation.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"r2": r2, "pearson": pearson}


# ------------------------- Lightning module -------------------------


class LinearHead(pl.LightningModule):
    """LayerNorm + Dropout + Linear head.

    - Classification (binary): 1 logit + BCEWithLogitsLoss; logs auroc/auprc/f1/balanced_accuracy/accuracy
    - Regression: 1 output + MSE; logs R2 + Pearson
    """

    def __init__(
        self,
        input_dim: int,
        lr: float = 1e-2,
        dropout: float = 0.1,
        regression: bool = False,
        pos_weight: float | None = None,  # optional for class imbalance
    ):
        """Initialize the LinearHead model.

        Parameters
        ----------
        input_dim : int
            Dimension of input features.
        lr : float, optional
            Learning rate, by default 1e-2.
        dropout : float, optional
            Dropout probability, by default 0.1.
        regression : bool, optional
            If True, use MSE loss for regression. If False, use BCEWithLogitsLoss for classification.
            By default False.
        pos_weight : float | None, optional
            Weight for the positive class in BCE loss (for imbalanced classification), by default None.
        """
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),  # <— always 1 output
        )
        self.lr = lr
        self.regression = regression

        # Optional positive-class weighting for BCE
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.pos_weight = None

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
            Output logits or predictions of shape (batch_size, 1).
        """
        return self.net(x)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss based on regression or classification.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits/values.
        y : torch.Tensor
            Target values.

        Returns
        -------
        torch.Tensor
            Computed loss (MSE or BCE).
        """
        if self.regression:
            # y shape: (B, 1)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            return F.mse_loss(logits, y)
        # classification: BCEWithLogits
        if y.ndim == 1:
            y = y.unsqueeze(1)
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

    def training_step(self, batch, batch_idx):
        """Training step: compute logits and loss, log the loss.

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
        """Validation step: compute logits and loss, store for metrics.

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
        """Compute and log validation metrics at the end of the epoch."""
        if not self._val_logits:
            return
        logits = torch.cat(self._val_logits, dim=0).numpy()
        targets = torch.cat(self._val_targets, dim=0).numpy()

        if self.regression:
            preds = logits.reshape(-1)
            targs = targets.reshape(-1)
            metrics = _regression_metrics(targs, preds)
            self.log("val_r2", metrics["r2"], prog_bar=True, on_epoch=True)
            self.log("val_pearson", metrics["pearson"], on_epoch=True)
        else:
            scores = torch.sigmoid(torch.from_numpy(logits)).numpy().reshape(-1)
            targs = targets.reshape(-1)
            metrics = _classification_metrics(targs, scores)
            if metrics is None:
                # ensure the monitored key exists to avoid Lightning errors
                self.log("val_auroc", -1.0, prog_bar=True, on_epoch=True)
            else:
                for k, v in metrics.items():
                    self.log(f"val_{k}", v, on_epoch=True, prog_bar=(k == "auroc"))
        self._val_logits.clear()
        self._val_targets.clear()

    def test_step(self, batch, batch_idx):
        """Test step: compute logits and loss, store for metrics.

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
        """Compute and log test metrics at the end of the epoch."""
        if not self._test_logits:
            return
        logits = torch.cat(self._test_logits, dim=0).numpy()
        targets = torch.cat(self._test_targets, dim=0).numpy()
        if self.regression:
            preds = logits.reshape(-1)
            targs = targets.reshape(-1)
            metrics = _regression_metrics(targs, preds)
        else:
            scores = torch.sigmoid(torch.from_numpy(logits)).numpy().reshape(-1)
            targs = targets.reshape(-1)
            metrics = _classification_metrics(targs, scores)  # may be None
        if metrics is not None:
            for k, v in metrics.items():
                self.log(f"test_{k}", v, on_epoch=True)
        self.test_results_ = metrics  # may be None
        self._test_logits.clear()
        self._test_targets.clear()

    def configure_optimizers(self):
        """Configure optimizer: AdamW with the specified learning rate.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ------------------------- main helpers -------------------------


def filter_drugs(
    df: pd.DataFrame,
    drug_cols: list[str],
    regression: bool,
    total_min_samples: int = 500,
    min_class_samples: int = 50,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop drug columns that don't meet minimum sample requirements.

    - Classification: require >= total_min_samples non-null AND at least 2 classes
      each with >= min_class_samples.
    - Regression: require >= total_min_samples non-null AND non-constant targets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing drug columns.
    drug_cols : list[str]
        List of drug column names to check.
    regression : bool
        Whether the task is regression.
    total_min_samples : int, optional
        Minimum total non-null samples required, by default 500.
    min_class_samples : int, optional
        Minimum samples per class (for classification), by default 50.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - Filtered dataframe (with only kept drug columns + other columns).
        - List of kept drug column names.
    """
    keep_cols = []
    for col in drug_cols:
        series = df[col].dropna()
        if len(series) < total_min_samples:
            continue
        if regression:
            if series.nunique() > 1:
                keep_cols.append(col)
        else:
            vc = series.value_counts()
            vc = {v: c for v, c in vc.items() if c >= min_class_samples}
            if len(vc) >= 2:
                keep_cols.append(col)
    base_cols = [c for c in df.columns if c not in drug_cols]
    kept = list(base_cols) + keep_cols
    kept = [c for c in kept if c in df.columns]
    return df[kept].copy(), keep_cols


def _make_loaders(
    X_train, y_train, X_val, y_val, X_test=None, y_test=None, batch_size: int = 256, num_workers: int = 4
):
    """Wrap numpy splits in TensorDatasets and DataLoaders for training/validation/test.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation targets.
    X_test : np.ndarray, optional
        Test features, by default None.
    y_test : np.ndarray, optional
        Test targets, by default None.
    batch_size : int, optional
        Batch size, by default 256.
    num_workers : int, optional
        Number of data loading workers, by default 4.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader | None]
        Training loader, validation loader, and optional test loader.
    """
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    if X_test is not None and y_test is not None:
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        # Test loader is optional (only created when both features and targets are supplied)
        test_dl = None
    return train_dl, val_dl, test_dl


def train_and_predict(
    df: pd.DataFrame,
    model_name: str,
    regression: bool,
    lr: float,
    drug: str,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    split: str = "random",
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
    regression : bool
        Whether to perform regression (True) or classification (False).
    lr : float
        Learning rate.
    drug : str
        Name of the drug (target column).
    max_epochs : int, optional
        Maximum training epochs, by default 100.
    early_stopping_patience : int, optional
        Patience for early stopping, by default 10.
    split : str, optional
        Split mode or column name, by default "random".
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
    if split is None or split == "random":
        cols = [model_name, drug]
    else:
        cols = [model_name, split, drug]

    sub = df[cols].dropna().reset_index(drop=True)
    if len(sub) == 0:
        return {"drug": drug, "seed": seed, "model_name": model_name, "skipped": "no_data"}

    # Features and labels
    X = _to_numpy_matrix(sub[model_name])
    if regression:
        # ensure a plain numpy ndarray of floats
        y = sub[drug].astype(float).to_numpy(dtype=np.float32).reshape(-1, 1)
    else:
        y_int, _ = _encode_labels(sub[drug])
        # BCE needs float targets in {0,1}; y_int is already ndarray
        y = y_int.astype(np.float32).reshape(-1, 1)

    input_dim = X.shape[1]

    # Split (robust with fallbacks)
    groups = (
        sub[split].astype(str).to_numpy()
        if (split is not None and split != "random" and split in sub.columns)
        else None
    )
    try:
        tr_idx, va_idx, te_idx = _split_indices(
            y=y.reshape(-1),
            groups=groups,
            split_mode=split if (split is not None and split != "random") else None,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            is_regression=regression,
        )
    except ValueError as e:
        return {
            "drug": drug,
            "seed": seed,
            "model_name": model_name,
            "split": split,
            "skipped": f"split_failed: {str(e)}",
        }

    Xtr, Xva, Xte = X[tr_idx], X[va_idx], X[te_idx]
    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

    train_dl, val_dl, test_dl = _make_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=256, num_workers=4)

    # Model
    lit_model = LinearHead(input_dim=input_dim, lr=lr, dropout=0.1, regression=regression)

    # Callbacks
    if regression:
        monitor = "val_r2"
        filename = f"{drug}-{{epoch:02d}}-{{val_r2:.4f}}"
    else:
        monitor = "val_auroc"
        filename = f"{drug}-{{epoch:02d}}-{{val_auroc:.4f}}"

    ckpt_cb = ModelCheckpoint(
        monitor=monitor,
        mode="max",
        save_top_k=1,
        save_last=False,
        filename=filename,
    )
    es_cb = EarlyStopping(monitor=monitor, mode="max", patience=early_stopping_patience, min_delta=0.0)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        deterministic=False,
        callbacks=[ckpt_cb, es_cb],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(lit_model, train_dl, val_dl)

    # Load best
    best_path = ckpt_cb.best_model_path
    if best_path and os.path.exists(best_path):
        lit_model = LinearHead.load_from_checkpoint(best_path)

    # Evaluate on validation (best checkpoint)
    lit_model.eval()
    device = next(lit_model.parameters()).device
    with torch.no_grad():
        logits_val = []
        y_val_all = []
        for xb, yb in val_dl:
            logits_val.append(lit_model(xb.to(device)).cpu())
            y_val_all.append(yb.cpu())
        logits_val = torch.cat(logits_val, 0).numpy().reshape(-1)
        y_val_all = torch.cat(y_val_all, 0).numpy().reshape(-1)

        if regression:
            y_pred = logits_val
            y_true = y_val_all
            val_metrics = _regression_metrics(y_true, y_pred)
        else:
            scores = torch.sigmoid(torch.from_numpy(logits_val)).numpy().reshape(-1)
            y_true = y_val_all.astype(int)
            val_metrics = _classification_metrics(y_true, scores)  # may be None

    # Build result dict
    result = {
        "drug": drug,
        "seed": seed,
        "model_name": model_name,
        "split": split,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "best_ckpt": best_path or "",
    }

    if regression:
        if val_metrics is None:
            result.update({"val_r2": None, "val_pearson": None})
        else:
            result.update({"val_r2": val_metrics["r2"], "val_pearson": val_metrics["pearson"]})
    else:
        if val_metrics is None:
            result.update(
                {
                    "val_auroc": None,
                    "val_auprc": None,
                    "val_f1": None,
                    "val_balanced_accuracy": None,
                    "val_accuracy": None,
                }
            )
        else:
            result.update(
                {
                    "val_auroc": val_metrics["auroc"],
                    "val_auprc": val_metrics["auprc"],
                    "val_f1": val_metrics["f1"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                }
            )

    if test_after_train and test_dl is not None:
        # Lightning test to reuse code paths
        trainer.test(lit_model, test_dl, verbose=False)
        device = next(lit_model.parameters()).device
        test_metrics = getattr(lit_model, "test_results_", None)
        if test_metrics is None:
            # Fallback manual
            logits_test = []
            y_test_all = []
            with torch.no_grad():
                for xb, yb in test_dl:
                    logits_test.append(lit_model(xb.to(device)).cpu())
                    y_test_all.append(yb.cpu())
            logits_test = torch.cat(logits_test, 0).numpy().reshape(-1)
            y_test_all = torch.cat(y_test_all, 0).numpy().reshape(-1)

            if regression:
                y_pred = logits_test
                y_true = y_test_all
                test_metrics = _regression_metrics(y_true, y_pred)
            else:
                scores = torch.sigmoid(torch.from_numpy(logits_test)).numpy().reshape(-1)
                y_true = y_test_all.astype(int)
                test_metrics = _classification_metrics(y_true, scores)  # may be None

        if regression:
            if test_metrics is None:
                result.update({"test_r2": None, "test_pearson": None})
            else:
                result.update({"test_r2": test_metrics["r2"], "test_pearson": test_metrics["pearson"]})
        else:
            if test_metrics is None:
                result.update(
                    {
                        "test_auroc": None,
                        "test_auprc": None,
                        "test_f1": None,
                        "test_balanced_accuracy": None,
                        "test_accuracy": None,
                    }
                )
            else:
                result.update(
                    {
                        "test_auroc": test_metrics["auroc"],
                        "test_auprc": test_metrics["auprc"],
                        "test_f1": test_metrics["f1"],
                        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                        "test_accuracy": test_metrics["accuracy"],
                    }
                )

    return result


def run(
    df: pd.DataFrame,
    drug_cols: list[str],
    model_name: str,
    lr: float,
    regression: bool = False,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    total_min_samples: int = 500,
    min_class_samples: int = 50,
    split: str = "genus",
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    test_after_train: bool = False,
    seeds: list[int] | None = None,
    limit_n_drugs: int | None = None,
):
    """Run the training and prediction for phenotypic traits.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    drug_cols : list[str]
        List of drug columns to process.
    model_name : str
        Name of the model (feature column).
    regression : bool, optional
        Whether to perform regression, by default False.
    lr : float
        Learning rate.
    max_epochs : int, optional
        Maximum training epochs, by default 100.
    early_stopping_patience : int, optional
        Patience for early stopping, by default 10.
    total_min_samples : int, optional
        Minimum total samples required for a drug, by default 500.
    min_class_samples : int, optional
        Minimum samples per class (for classification), by default 50.
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
    limit_n_drugs : int | None, optional
        Limit number of drugs to process (for debugging), by default None.

    Returns
    -------
    pd.DataFrame
        Dataframe containing aggregated results for all drugs and seeds.
    """
    if seeds is None:
        seeds = [1]
    filtered_df, drug_cols = filter_drugs(df, drug_cols, regression, total_min_samples, min_class_samples)
    if limit_n_drugs is not None:
        drug_cols = drug_cols[:limit_n_drugs]

    out = []
    for seed in seeds:
        for drug in tqdm(drug_cols):
            res = train_and_predict(
                filtered_df,
                model_name=model_name,
                regression=regression,
                lr=lr,
                drug=drug,
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

    # ------------------------------------------------------------
    # Exclude failed drugs from aggregation
    # ------------------------------------------------------------
    total_rows = len(out_df)
    failed_mask = out_df["skipped"].notna() if "skipped" in out_df.columns else pd.Series(False, index=out_df.index)

    # Choose primary metric used to decide validity for aggregation
    primary_val = "val_r2" if regression else "val_auroc"
    primary_test = "test_r2" if regression else "test_auroc"

    # Validation subset: not failed AND primary metric present and not null
    if primary_val in out_df.columns:
        valid_val_mask = (~failed_mask) & out_df[primary_val].notna()
    else:
        valid_val_mask = (~failed_mask) & False  # no primary metric column -> nothing valid

    val_df = out_df[valid_val_mask].copy()
    n_val_used = len(val_df)

    # Report mean metrics across drugs and seeds (validation metrics)
    if regression:
        pref = ["val_r2", "val_pearson"]
    else:
        pref = ["val_auroc", "val_auprc", "val_f1", "val_balanced_accuracy", "val_accuracy"]

    available = [m for m in pref if m in val_df.columns]
    if n_val_used > 0 and available:
        means = val_df[available].mean(numeric_only=True)
        print(f"\n=== Mean validation metrics across drugs/seeds (included {n_val_used} of {total_rows}) ===")
        for k, v in means.items():
            print(f"{k}: {v:.4f}")
    else:
        print(f"\n=== No valid validation rows to aggregate (included 0 of {total_rows}) ===")

    # Test subset (only if requested): use same failure filter + primary test metric not null
    if test_after_train:
        if primary_test in out_df.columns:
            valid_test_mask = (~failed_mask) & out_df[primary_test].notna()
        else:
            valid_test_mask = (~failed_mask) & False

        test_df = out_df[valid_test_mask].copy()
        n_test_used = len(test_df)

        if regression:
            pref_test = ["test_r2", "test_pearson"]
        else:
            pref_test = ["test_auroc", "test_auprc", "test_f1", "test_balanced_accuracy", "test_accuracy"]

        available_test = [m for m in pref_test if m in test_df.columns]
        if n_test_used > 0 and available_test:
            means_test = test_df[available_test].mean(numeric_only=True)
            print(f"\n=== Mean test metrics across drugs/seeds (included {n_test_used} of {total_rows}) ===")
            for k, v in means_test.items():
                print(f"{k}: {v:.4f}")
        else:
            print(f"\n=== No valid test rows to aggregate (included 0 of {total_rows}) ===")

    return out_df


# ------------------------- CLI -------------------------


class ArgParser(Tap):
    """Arguments for drugs linear probing."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_genomes_df_filepath: str
    labels_df_filepath: str
    output_dir: str
    lr: float
    model_name: str  # column name for model features
    regression: bool = False  # if True, use regression loss instead of classification
    max_epochs: int = 100
    early_stopping_patience: int = 10
    total_min_samples: int = 500  # minimum total samples for a drug to be considered
    min_class_samples: int = 50
    split: str = "random"
    train_size: float = 0.7
    val_size: float = 0.1
    test_size: float = 0.2
    test_after_train: bool = False
    limit_n_drugs: int | None = None  # limit number of drugs to process, for debugging


if __name__ == "__main__":
    args = ArgParser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.input_genomes_df_filepath)
    labels_df = pd.read_csv(args.labels_df_filepath)
    drug_cols = list(labels_df.columns[1:])
    df = pd.merge(df, labels_df, on="genome_name", how="inner")

    today = datetime.today().strftime("%Y_%m_%d")
    print(f"\nRunning AMR prediction for model: {args.model_name}")
    metrics_df = run(
        df=df,
        drug_cols=drug_cols,
        model_name=args.model_name,
        lr=args.lr,
        regression=args.regression,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        total_min_samples=args.total_min_samples,
        min_class_samples=args.min_class_samples,
        split=args.split,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        test_after_train=args.test_after_train,
        seeds=[1, 2, 3],
        limit_n_drugs=args.limit_n_drugs,
    )
    out_path = os.path.join(
        args.output_dir, f"amr_preds_regression_{args.regression}_split_{args.split}_{args.model_name}_{today}.csv"
    )
    metrics_df.to_csv(out_path, index=False)
    print(f"\nSaved metrics to: {out_path}")
