from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupShuffleSplit
from tap import Tap

if __package__:
    from .train_and_predict_linear import LinearHead, _classification_metrics, _make_loaders
else:
    from train_and_predict_linear import LinearHead, _classification_metrics, _make_loaders


DEFAULT_INPUT_FILEPATH = (
    "/home/mw896/rds/rds-flotolab-9X9gY1OFt4M/projects/bacformer/"
    "input-data/datasets/amr/models/all_model_embeddings_with_amr_and_evo.parquet"
)
DEFAULT_LABELS_FILEPATH = (
    "/home/mw896/rds/rds-flotolab-9X9gY1OFt4M/projects/bacformer/input-data/datasets/amr/binary_labels.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/mw896/rds/rds-flotolab-9X9gY1OFt4M/projects/bacformer/"
    "input-data/datasets/amr/models/genus_split_linear_results"
)
DEFAULT_LEARNING_RATES = [0.05, 0.01, 0.005, 0.001, 0.0001]
DEFAULT_SEEDS = [1, 2, 3]
METADATA_COLUMNS = {"genome_name", "species", "genus", "family"}


@dataclass(frozen=True)
class DrugData:
    """Labels and row positions for one eligible antibiotic."""

    drug: str
    row_indices: np.ndarray
    y: np.ndarray
    groups: np.ndarray


@dataclass(frozen=True)
class GenusSplit:
    """Indices and provenance for a strict genus-disjoint split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    random_state: int
    fingerprint: str


def select_model_names(schema_columns: list[str], requested_models: list[str] | None = None) -> list[str]:
    """Select embedding columns from the parquet schema."""
    inferred = [column for column in schema_columns if column not in METADATA_COLUMNS]
    if requested_models is None:
        if not inferred:
            raise ValueError("No embedding model columns were found in the input parquet.")
        return inferred

    requested_models = list(dict.fromkeys(requested_models))
    missing = [model for model in requested_models if model not in schema_columns]
    if missing:
        raise ValueError(f"Requested model columns are missing from the parquet: {missing}")
    metadata_requested = [model for model in requested_models if model in METADATA_COLUMNS]
    if metadata_requested:
        raise ValueError(f"Metadata columns cannot be evaluated as models: {metadata_requested}")
    return requested_models


def read_labels(labels_filepath: str) -> tuple[pd.DataFrame, list[str]]:
    """Read and validate the wide binary AMR label table."""
    labels_df = pd.read_csv(labels_filepath)
    if "genome_name" not in labels_df.columns:
        raise ValueError("The labels CSV must contain a 'genome_name' column.")
    if labels_df["genome_name"].isna().any():
        raise ValueError("The labels CSV contains missing genome names.")
    duplicated = labels_df.loc[labels_df["genome_name"].duplicated(), "genome_name"]
    if not duplicated.empty:
        examples = duplicated.astype(str).head(5).tolist()
        raise ValueError(f"The labels CSV contains duplicate genome names, for example: {examples}")

    drug_columns = [column for column in labels_df.columns if column != "genome_name"]
    if not drug_columns:
        raise ValueError("The labels CSV does not contain any antibiotic columns.")

    for drug in drug_columns:
        non_null = labels_df[drug].dropna()
        if non_null.empty:
            continue
        numeric = pd.to_numeric(non_null, errors="raise")
        observed = set(numeric.astype(float).unique())
        if not observed.issubset({0.0, 1.0}):
            raise ValueError(f"Drug {drug!r} contains labels outside binary 0/1: {sorted(observed)}")
        labels_df.loc[non_null.index, drug] = numeric.astype(np.float32)

    return labels_df, drug_columns


def _stack_embeddings(series: pd.Series, genome_names: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Stack valid embeddings and return their original row positions."""
    arrays = []
    valid_positions = []
    expected_dim = None

    for position, (genome_name, value) in enumerate(zip(genome_names, series, strict=True)):
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):  # noqa
            continue
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise ValueError(f"Genome {genome_name!r} has an empty embedding.")
        if not np.isfinite(array).all():
            raise ValueError(f"Genome {genome_name!r} has a non-finite embedding.")
        if expected_dim is None:
            expected_dim = array.size
        elif array.size != expected_dim:
            raise ValueError(f"Genome {genome_name!r} has embedding dimension {array.size}; expected {expected_dim}.")
        arrays.append(array)
        valid_positions.append(position)

    if not arrays:
        raise ValueError("The model column does not contain any valid embeddings.")
    return np.stack(arrays), np.asarray(valid_positions, dtype=np.int64)


def read_model_dataset(
    input_filepath: str,
    labels_df: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Read one embedding column and align it with taxonomy and AMR labels."""
    model_df = pd.read_parquet(input_filepath, columns=["genome_name", "genus", model_name])
    if model_df["genome_name"].isna().any():
        raise ValueError("The embeddings parquet contains missing genome names.")
    duplicated = model_df.loc[model_df["genome_name"].duplicated(), "genome_name"]
    if not duplicated.empty:
        examples = duplicated.astype(str).head(5).tolist()
        raise ValueError(f"The embeddings parquet contains duplicate genome names, for example: {examples}")

    embeddings, valid_positions = _stack_embeddings(model_df[model_name], model_df["genome_name"])
    model_df = model_df.iloc[valid_positions][["genome_name", "genus"]].copy()
    model_df["_embedding_index"] = np.arange(len(model_df), dtype=np.int64)
    model_df = model_df.dropna(subset=["genome_name", "genus"])
    model_df = model_df[model_df["genus"].astype(str).str.strip().ne("")]

    merged = model_df.merge(labels_df, on="genome_name", how="inner", validate="one_to_one", sort=False)
    if merged.empty:
        raise ValueError(f"Model {model_name!r} has no genomes matching the labels CSV.")
    embedding_indices = merged.pop("_embedding_index").to_numpy(dtype=np.int64)
    return merged.reset_index(drop=True), embeddings[embedding_indices]


def prepare_drugs(
    model_df: pd.DataFrame,
    drug_columns: list[str],
    total_min_samples: int,
    min_class_samples: int,
    limit_n_drugs: int | None = None,
) -> dict[str, DrugData]:
    """Prepare drugs that satisfy the existing AMR support thresholds."""
    prepared = {}
    for drug in drug_columns:
        mask = model_df[drug].notna().to_numpy()
        row_indices = np.flatnonzero(mask)
        if len(row_indices) < total_min_samples:
            continue

        y = pd.to_numeric(model_df.loc[mask, drug], errors="raise").to_numpy(dtype=np.float32)
        observed = set(np.unique(y).astype(float))
        if not observed.issubset({0.0, 1.0}):
            raise ValueError(f"Drug {drug!r} contains labels outside binary 0/1: {sorted(observed)}")
        class_counts = pd.Series(y).value_counts()
        if set(class_counts.index.astype(float)) != {0.0, 1.0} or int(class_counts.min()) < min_class_samples:
            continue

        prepared[drug] = DrugData(
            drug=drug,
            row_indices=row_indices,
            y=y,
            groups=model_df.loc[mask, "genus"].astype(str).to_numpy(),
        )
        if limit_n_drugs is not None and len(prepared) >= limit_n_drugs:
            break

    return prepared


def _split_fingerprint(groups: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> str:
    payload = {
        "train": sorted(set(groups[train].tolist())),
        "val": sorted(set(groups[val].tolist())),
        "test": sorted(set(groups[test].tolist())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def split_by_genus(
    y: np.ndarray,
    groups: np.ndarray,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    max_attempts: int = 100,
) -> GenusSplit:
    """Create a strict genus-disjoint split with both classes in every partition."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Train, validation, and test proportions must sum to 1.0.")
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Train, validation, and test proportions must all be positive.")
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three genera are required for a strict train/validation/test split.")
    if len(np.unique(y)) != 2:
        raise ValueError("Both binary classes are required before genus splitting.")

    all_indices = np.arange(len(y))
    val_fraction_of_remainder = val_size / (val_size + test_size)
    for attempt in range(max_attempts):
        random_state = seed + attempt * 1009
        try:
            first_split = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
            train, remainder = next(first_split.split(all_indices, y, groups=groups))
            second_split = GroupShuffleSplit(
                n_splits=1,
                train_size=val_fraction_of_remainder,
                random_state=random_state + 1,
            )
            val_relative, test_relative = next(second_split.split(remainder, y[remainder], groups=groups[remainder]))
        except ValueError:
            continue

        val = remainder[val_relative]
        test = remainder[test_relative]
        if any(len(np.unique(y[indices])) < 2 for indices in (train, val, test)):
            continue

        train_groups = set(groups[train])
        val_groups = set(groups[val])
        test_groups = set(groups[test])
        if not (
            train_groups.isdisjoint(val_groups)
            and train_groups.isdisjoint(test_groups)
            and val_groups.isdisjoint(test_groups)
        ):
            raise RuntimeError("Internal error: genus leakage was detected after GroupShuffleSplit.")

        return GenusSplit(
            train=train,
            val=val,
            test=test,
            random_state=random_state,
            fingerprint=_split_fingerprint(groups, train, val, test),
        )

    raise ValueError(
        f"Could not find a genus split containing both classes in every partition after {max_attempts} attempts."
    )


def _evaluate(model: LinearHead, data_loader) -> dict[str, float]:
    """Evaluate a fitted binary linear probe."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logits = []
    targets = []
    with torch.inference_mode():
        for features, labels in data_loader:
            logits.append(model(features.to(device)).cpu())
            targets.append(labels.cpu())

    scores = torch.sigmoid(torch.cat(logits).reshape(-1)).numpy()
    y_true = torch.cat(targets).reshape(-1).numpy().astype(int)
    metrics = _classification_metrics(y_true, scores)
    if metrics is None:
        raise RuntimeError("A supposedly valid evaluation partition contains only one class.")
    return metrics


def train_linear_probe(
    embeddings: np.ndarray,
    drug_data: DrugData,
    split: GenusSplit,
    lr: float,
    seed: int,
    max_epochs: int,
    early_stopping_patience: int,
    dropout: float,
    evaluate_test: bool,
    enable_progress_bar: bool,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Fit one binary linear probe and evaluate its best validation-AUPRC checkpoint."""
    pl.seed_everything(seed, workers=True)
    task_embeddings = embeddings[drug_data.row_indices]
    y = drug_data.y.reshape(-1, 1).astype(np.float32)
    train_loader, val_loader, test_loader = _make_loaders(
        task_embeddings[split.train],
        y[split.train],
        task_embeddings[split.val],
        y[split.val],
        task_embeddings[split.test],
        y[split.test],
        batch_size=256,
        num_workers=4,
    )

    model = LinearHead(
        input_dim=task_embeddings.shape[1],
        lr=lr,
        dropout=dropout,
        regression=False,
    )
    with tempfile.TemporaryDirectory(prefix="bacbench_amr_checkpoint_") as checkpoint_dir:
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val_auprc",
            mode="max",
            save_top_k=1,
            save_last=False,
            filename="best-{epoch:02d}-{val_auprc:.4f}",
        )
        early_stopping = EarlyStopping(
            monitor="val_auprc",
            mode="max",
            patience=early_stopping_patience,
            min_delta=0.0,
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            deterministic=False,
            callbacks=[checkpoint, early_stopping],
            default_root_dir=checkpoint_dir,
            logger=False,
            enable_checkpointing=True,
            enable_model_summary=False,
            enable_progress_bar=enable_progress_bar,
            log_every_n_steps=10,
        )
        trainer.fit(model, train_loader, val_loader)
        if not checkpoint.best_model_path:
            raise RuntimeError("Training completed without producing a best validation-AUPRC checkpoint.")
        best_model = LinearHead.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")
        val_metrics = _evaluate(best_model, val_loader)
        test_metrics = _evaluate(best_model, test_loader) if evaluate_test and test_loader is not None else None

    return val_metrics, test_metrics


def _result_metadata(
    model_name: str,
    drug_data: DrugData,
    split: GenusSplit,
    lr: float,
    seed: int,
) -> dict[str, object]:
    y = drug_data.y
    groups = drug_data.groups
    return {
        "model_name": model_name,
        "drug": drug_data.drug,
        "seed": seed,
        "lr": lr,
        "split": "genus",
        "split_random_state": split.random_state,
        "split_fingerprint": split.fingerprint,
        "n_total": len(y),
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "n_train_genera": len(np.unique(groups[split.train])),
        "n_val_genera": len(np.unique(groups[split.val])),
        "n_test_genera": len(np.unique(groups[split.test])),
        "train_positive_fraction": float(y[split.train].mean()),
        "val_positive_fraction": float(y[split.val].mean()),
        "test_positive_fraction": float(y[split.test].mean()),
    }


def run_lr_sweep(
    model_name: str,
    embeddings: np.ndarray,
    drugs: dict[str, DrugData],
    learning_rates: list[float],
    tuning_seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
    max_split_attempts: int,
    max_epochs: int,
    early_stopping_patience: int,
    dropout: float,
    enable_progress_bar: bool,
) -> pd.DataFrame:
    """Tune one model-wide learning rate on mean per-drug validation AUPRC."""
    splits = {}
    split_errors = {}
    for drug, drug_data in drugs.items():
        try:
            splits[drug] = split_by_genus(
                drug_data.y,
                drug_data.groups,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                seed=tuning_seed,
                max_attempts=max_split_attempts,
            )
        except ValueError as error:
            split_errors[drug] = str(error)

    results = []
    for lr in learning_rates:
        print(f"  Tuning {model_name} at lr={lr:g}")
        for drug, drug_data in drugs.items():
            if drug in split_errors:
                results.append(
                    {
                        "model_name": model_name,
                        "drug": drug,
                        "seed": tuning_seed,
                        "lr": lr,
                        "split": "genus",
                        "skipped": f"split_failed: {split_errors[drug]}",
                    }
                )
                continue
            split = splits[drug]
            val_metrics, _ = train_linear_probe(
                embeddings=embeddings,
                drug_data=drug_data,
                split=split,
                lr=lr,
                seed=tuning_seed,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                dropout=dropout,
                evaluate_test=False,
                enable_progress_bar=enable_progress_bar,
            )
            row = _result_metadata(model_name, drug_data, split, lr, tuning_seed)
            row.update({f"val_{metric}": value for metric, value in val_metrics.items()})
            row["skipped"] = None
            results.append(row)
    return pd.DataFrame(results)


def select_best_learning_rate(tuning_results: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Select the LR with highest mean validation AUPRC, preferring a smaller LR on ties."""
    if "val_auprc" not in tuning_results.columns:
        raise ValueError("No valid validation AUPRC values were produced during LR tuning.")
    valid = tuning_results[tuning_results["val_auprc"].notna()].copy()
    if "skipped" in valid.columns:
        valid = valid[valid["skipped"].isna()]
    if valid.empty:
        raise ValueError("No valid validation AUPRC values were produced during LR tuning.")

    summary = (
        valid.groupby(["model_name", "lr"], as_index=False)
        .agg(mean_val_auprc=("val_auprc", "mean"), n_drugs=("drug", "nunique"))
        .sort_values(["mean_val_auprc", "lr"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best_lr = float(summary.iloc[0]["lr"])
    summary["selected"] = np.isclose(summary["lr"], best_lr)
    return best_lr, summary


def run_final_evaluation(
    model_name: str,
    embeddings: np.ndarray,
    drugs: dict[str, DrugData],
    best_lr: float,
    seeds: list[int],
    train_size: float,
    val_size: float,
    test_size: float,
    max_split_attempts: int,
    max_epochs: int,
    early_stopping_patience: int,
    dropout: float,
    enable_progress_bar: bool,
) -> pd.DataFrame:
    """Train and test one model-wide selected LR on several genus-split seeds."""
    results = []
    for seed in seeds:
        print(f"  Evaluating {model_name} with lr={best_lr:g}, seed={seed}")
        for drug, drug_data in drugs.items():
            try:
                split = split_by_genus(
                    drug_data.y,
                    drug_data.groups,
                    train_size=train_size,
                    val_size=val_size,
                    test_size=test_size,
                    seed=seed,
                    max_attempts=max_split_attempts,
                )
            except ValueError as error:
                results.append(
                    {
                        "model_name": model_name,
                        "drug": drug,
                        "seed": seed,
                        "lr": best_lr,
                        "best_lr": best_lr,
                        "split": "genus",
                        "skipped": f"split_failed: {error}",
                    }
                )
                continue

            val_metrics, test_metrics = train_linear_probe(
                embeddings=embeddings,
                drug_data=drug_data,
                split=split,
                lr=best_lr,
                seed=seed,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                dropout=dropout,
                evaluate_test=True,
                enable_progress_bar=enable_progress_bar,
            )
            row = _result_metadata(model_name, drug_data, split, best_lr, seed)
            row["best_lr"] = best_lr
            row.update({f"val_{metric}": value for metric, value in val_metrics.items()})
            row.update({f"test_{metric}": value for metric, value in test_metrics.items()})
            row["skipped"] = None
            results.append(row)
    return pd.DataFrame(results)


def summarize_test_results(test_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid test rows by model for concise reporting."""
    if "test_auprc" not in test_results.columns:
        return pd.DataFrame()
    valid = test_results[test_results["test_auprc"].notna()].copy()
    if "skipped" in valid.columns:
        valid = valid[valid["skipped"].isna()]
    if valid.empty:
        return pd.DataFrame()

    metrics = ["test_auroc", "test_auprc", "test_f1", "test_balanced_accuracy", "test_accuracy"]
    summary = valid.groupby("model_name")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index()
    counts = valid.groupby("model_name", as_index=False).agg(
        best_lr=("best_lr", "first"),
        n_rows=("drug", "size"),
        n_drugs=("drug", "nunique"),
        n_seeds=("seed", "nunique"),
    )
    return counts.merge(summary, on="model_name", how="left", validate="one_to_one")


def _save_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)


class ArgParser(Tap):
    """Arguments for multi-model binary AMR evaluation on genus splits."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_genomes_df_filepath: str = DEFAULT_INPUT_FILEPATH
    labels_df_filepath: str = DEFAULT_LABELS_FILEPATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    model_names: list[str] | None = None
    learning_rates: list[float] = DEFAULT_LEARNING_RATES
    tuning_seed: int = 1
    seeds: list[int] = DEFAULT_SEEDS
    max_epochs: int = 100
    early_stopping_patience: int = 10
    total_min_samples: int = 500
    min_class_samples: int = 50
    train_size: float = 0.7
    val_size: float = 0.1
    test_size: float = 0.2
    dropout: float = 0.1
    max_split_attempts: int = 100
    limit_n_models: int | None = None
    limit_n_drugs: int | None = None
    enable_progress_bar: bool = False


def main(args: ArgParser) -> None:
    """Run LR tuning and final multi-seed evaluation for every requested model."""
    if not args.learning_rates:
        raise ValueError("At least one learning rate is required.")
    if not args.seeds:
        raise ValueError("At least one final evaluation seed is required.")

    labels_df, drug_columns = read_labels(args.labels_df_filepath)
    schema_columns = pq.ParquetFile(args.input_genomes_df_filepath).schema_arrow.names
    required_columns = METADATA_COLUMNS.intersection({"genome_name", "genus"})
    missing_required = sorted(required_columns.difference(schema_columns))
    if missing_required:
        raise ValueError(f"The embeddings parquet is missing required columns: {missing_required}")
    model_names = select_model_names(schema_columns, args.model_names)
    if args.limit_n_models is not None:
        model_names = model_names[: args.limit_n_models]

    run_id = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_dir = Path(args.output_dir)
    tuning_path = output_dir / f"amr_genus_lr_sweep_{run_id}.csv"
    tuning_summary_path = output_dir / f"amr_genus_lr_summary_{run_id}.csv"
    test_path = output_dir / f"amr_genus_test_results_{run_id}.csv"
    test_summary_path = output_dir / f"amr_genus_test_summary_{run_id}.csv"

    all_tuning_results = []
    all_tuning_summaries = []
    all_test_results = []
    for model_index, model_name in enumerate(model_names, start=1):
        print(f"\n[{model_index}/{len(model_names)}] Loading and evaluating model: {model_name}")
        model_df, embeddings = read_model_dataset(
            input_filepath=args.input_genomes_df_filepath,
            labels_df=labels_df,
            model_name=model_name,
        )
        drugs = prepare_drugs(
            model_df=model_df,
            drug_columns=drug_columns,
            total_min_samples=args.total_min_samples,
            min_class_samples=args.min_class_samples,
            limit_n_drugs=args.limit_n_drugs,
        )
        if not drugs:
            raise ValueError(f"Model {model_name!r} has no drugs meeting the configured support thresholds.")
        print(f"  Matched {len(model_df)} genomes; evaluating {len(drugs)} eligible drugs")

        tuning_results = run_lr_sweep(
            model_name=model_name,
            embeddings=embeddings,
            drugs=drugs,
            learning_rates=args.learning_rates,
            tuning_seed=args.tuning_seed,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            max_split_attempts=args.max_split_attempts,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience,
            dropout=args.dropout,
            enable_progress_bar=args.enable_progress_bar,
        )
        best_lr, tuning_summary = select_best_learning_rate(tuning_results)
        print(f"  Selected lr={best_lr:g} by mean validation AUPRC")
        all_tuning_results.append(tuning_results)
        all_tuning_summaries.append(tuning_summary)
        _save_csv(pd.concat(all_tuning_results, ignore_index=True), tuning_path)
        _save_csv(pd.concat(all_tuning_summaries, ignore_index=True), tuning_summary_path)

        test_results = run_final_evaluation(
            model_name=model_name,
            embeddings=embeddings,
            drugs=drugs,
            best_lr=best_lr,
            seeds=args.seeds,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            max_split_attempts=args.max_split_attempts,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience,
            dropout=args.dropout,
            enable_progress_bar=args.enable_progress_bar,
        )
        all_test_results.append(test_results)

        combined_tuning = pd.concat(all_tuning_results, ignore_index=True)
        combined_tuning_summary = pd.concat(all_tuning_summaries, ignore_index=True)
        combined_test = pd.concat(all_test_results, ignore_index=True)
        _save_csv(combined_tuning, tuning_path)
        _save_csv(combined_tuning_summary, tuning_summary_path)
        _save_csv(combined_test, test_path)
        _save_csv(summarize_test_results(combined_test), test_summary_path)

        if "test_auprc" in test_results.columns:
            model_valid = test_results[test_results["test_auprc"].notna()]
        else:
            model_valid = pd.DataFrame()
        if model_valid.empty:
            print("  No valid test runs were produced for this model")
        else:
            print(
                f"  Mean test AUPRC={model_valid['test_auprc'].mean():.4f}; "
                f"mean test AUROC={model_valid['test_auroc'].mean():.4f} "
                f"across {len(model_valid)} drug/seed runs"
            )

    print(f"\nSaved detailed test results to: {test_path}")
    print(f"Saved model-level test summary to: {test_summary_path}")
    print(f"Saved detailed LR sweep results to: {tuning_path}")
    print(f"Saved LR sweep summary to: {tuning_summary_path}")


if __name__ == "__main__":
    main(ArgParser().parse_args())
