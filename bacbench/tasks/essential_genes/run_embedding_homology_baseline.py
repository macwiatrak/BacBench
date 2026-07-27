from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

INPUT_FILEPATH = Path("/Users/maciejwiatrak/Downloads/esmc_embeds_flat.parquet")
PER_GENOME_OUTPUT_FILEPATH = Path("/Users/maciejwiatrak/Downloads/esmc_homology_baseline_per_genome.csv")
SUMMARY_OUTPUT_FILEPATH = Path("/Users/maciejwiatrak/Downloads/esmc_homology_baseline_summary.csv")
K_TUNING_OUTPUT_FILEPATH = Path("/Users/maciejwiatrak/Downloads/esmc_homology_baseline_k_tuning.csv")

PARQUET_BATCH_SIZE = 2048
QUERY_BATCH_SIZE = 64
K_VALUES = (1, 3, 5, 10)
REQUIRED_COLUMNS = {"genome_name", "embeddings", "split", "essential"}


@dataclass(frozen=True)
class DatasetLayout:
    """Validated sizes needed to allocate the embedding matrices."""

    embedding_dim: int
    n_train_essential: int
    n_train_nonessential: int
    n_validation: int
    n_test: int


@dataclass(frozen=True)
class HomologyData:
    """Normalized reference and query data for exact cosine search."""

    train_essential: np.ndarray
    train_nonessential: np.ndarray
    validation_embeddings: np.ndarray
    validation_labels: np.ndarray
    validation_genomes: np.ndarray
    test_embeddings: np.ndarray
    test_labels: np.ndarray
    test_genomes: np.ndarray


@dataclass(frozen=True)
class ClassConditionalScores:
    """Continuous margin and its class-conditional similarity components."""

    margin: np.ndarray
    essential_similarity: np.ndarray
    nonessential_similarity: np.ndarray


def inspect_dataset(input_filepath: Path) -> DatasetLayout:
    """Validate metadata and determine the required array sizes."""
    parquet_file = pq.ParquetFile(input_filepath)
    missing_columns = REQUIRED_COLUMNS.difference(parquet_file.schema_arrow.names)
    if missing_columns:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing_columns)}")

    metadata = pd.read_parquet(input_filepath, columns=["genome_name", "split", "essential"])
    if metadata[["genome_name", "split", "essential"]].isna().any().any():
        raise ValueError("Genome names, splits, and essential labels must not contain missing values.")
    observed_labels = set(metadata["essential"].unique())
    if not observed_labels.issubset({0, 1}):
        raise ValueError(f"Essential labels must be binary 0/1; observed {sorted(observed_labels)}")
    if not {"train", "val", "test"}.issubset(set(metadata["split"].unique())):
        raise ValueError("Input parquet must contain train, val, and test rows.")

    splits_per_genome = metadata.groupby("genome_name")["split"].nunique()
    leaking_genomes = splits_per_genome[splits_per_genome > 1]
    if not leaking_genomes.empty:
        examples = leaking_genomes.index.astype(str).tolist()[:5]
        raise ValueError(f"Genomes occur in more than one split, for example: {examples}")

    first_batch = next(parquet_file.iter_batches(columns=["embeddings"], batch_size=1))
    first_embedding = first_batch.column(0)[0].as_py()
    if first_embedding is None or len(first_embedding) == 0:
        raise ValueError("Could not determine a valid embedding dimension from the first row.")
    embedding_dim = len(first_embedding)

    train_mask = metadata["split"].eq("train")
    validation_mask = metadata["split"].eq("val")
    test_mask = metadata["split"].eq("test")
    layout = DatasetLayout(
        embedding_dim=embedding_dim,
        n_train_essential=int((train_mask & metadata["essential"].eq(1)).sum()),
        n_train_nonessential=int((train_mask & metadata["essential"].eq(0)).sum()),
        n_validation=int(validation_mask.sum()),
        n_test=int(test_mask.sum()),
    )
    if min(layout.n_train_essential, layout.n_train_nonessential, layout.n_validation, layout.n_test) == 0:
        raise ValueError("Train must contain both classes and validation/test must contain proteins.")

    split_summary = metadata.groupby("split").agg(
        proteins=("essential", "size"),
        genomes=("genome_name", "nunique"),
        essential=("essential", "sum"),
        essential_fraction=("essential", "mean"),
    )
    print("Dataset summary:")
    print(split_summary.to_string())
    print(f"Embedding dimension: {embedding_dim}")
    return layout


def _normalize_rows_in_place(matrix: np.ndarray, name: str) -> None:
    """L2-normalize a matrix in place for cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError(f"{name} contains non-finite embeddings.")
    if np.any(norms == 0):
        raise ValueError(f"{name} contains zero-norm embeddings.")
    matrix /= norms[:, None]


def load_homology_data(input_filepath: Path, layout: DatasetLayout) -> HomologyData:
    """Load train references plus validation and test queries into CPU arrays."""
    train_essential = np.empty(
        (layout.n_train_essential, layout.embedding_dim),
        dtype=np.float32,
    )
    train_nonessential = np.empty(
        (layout.n_train_nonessential, layout.embedding_dim),
        dtype=np.float32,
    )
    validation_embeddings = np.empty((layout.n_validation, layout.embedding_dim), dtype=np.float32)
    validation_labels = np.empty(layout.n_validation, dtype=np.int8)
    validation_genomes = np.empty(layout.n_validation, dtype=object)
    test_embeddings = np.empty((layout.n_test, layout.embedding_dim), dtype=np.float32)
    test_labels = np.empty(layout.n_test, dtype=np.int8)
    test_genomes = np.empty(layout.n_test, dtype=object)

    essential_cursor = 0
    nonessential_cursor = 0
    validation_cursor = 0
    test_cursor = 0
    parquet_file = pq.ParquetFile(input_filepath)
    total_batches = math.ceil(parquet_file.metadata.num_rows / PARQUET_BATCH_SIZE)
    batches = parquet_file.iter_batches(
        columns=["embeddings", "split", "essential", "genome_name"],
        batch_size=PARQUET_BATCH_SIZE,
    )

    for batch in tqdm(batches, total=total_batches, desc="Loading embeddings", unit="batch"):
        embedding_column = batch.column(0)
        if embedding_column.null_count:
            raise ValueError("Embedding column contains missing values.")
        lengths = pc.list_value_length(embedding_column).to_numpy(zero_copy_only=False)
        if not np.all(lengths == layout.embedding_dim):
            raise ValueError("Embedding dimensions are inconsistent.")

        flat_embeddings = embedding_column.values.to_numpy(zero_copy_only=False).astype(
            np.float32,
            copy=False,
        )
        embeddings = flat_embeddings.reshape(len(batch), layout.embedding_dim)
        splits = np.asarray(batch.column(1).to_pylist(), dtype=object)
        labels = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        genomes = np.asarray(batch.column(3).to_pylist(), dtype=object)

        essential_mask = (splits == "train") & (labels == 1)
        n_essential = int(essential_mask.sum())
        train_essential[essential_cursor : essential_cursor + n_essential] = embeddings[essential_mask]
        essential_cursor += n_essential

        nonessential_mask = (splits == "train") & (labels == 0)
        n_nonessential = int(nonessential_mask.sum())
        train_nonessential[nonessential_cursor : nonessential_cursor + n_nonessential] = embeddings[nonessential_mask]
        nonessential_cursor += n_nonessential

        validation_mask = splits == "val"
        n_validation = int(validation_mask.sum())
        validation_embeddings[validation_cursor : validation_cursor + n_validation] = embeddings[validation_mask]
        validation_labels[validation_cursor : validation_cursor + n_validation] = labels[validation_mask]
        validation_genomes[validation_cursor : validation_cursor + n_validation] = genomes[validation_mask]
        validation_cursor += n_validation

        test_mask = splits == "test"
        n_test = int(test_mask.sum())
        test_embeddings[test_cursor : test_cursor + n_test] = embeddings[test_mask]
        test_labels[test_cursor : test_cursor + n_test] = labels[test_mask]
        test_genomes[test_cursor : test_cursor + n_test] = genomes[test_mask]
        test_cursor += n_test

    observed_cursors = (essential_cursor, nonessential_cursor, validation_cursor, test_cursor)
    expected_cursors = (
        layout.n_train_essential,
        layout.n_train_nonessential,
        layout.n_validation,
        layout.n_test,
    )
    if observed_cursors != expected_cursors:
        raise RuntimeError(f"Loaded row counts {observed_cursors} do not match expected counts {expected_cursors}.")

    print("Normalizing embeddings for exact cosine search...")
    _normalize_rows_in_place(train_essential, "Train-essential reference")
    _normalize_rows_in_place(train_nonessential, "Train-nonessential reference")
    _normalize_rows_in_place(validation_embeddings, "Validation queries")
    _normalize_rows_in_place(test_embeddings, "Test queries")

    allocated_gib = (
        train_essential.nbytes + train_nonessential.nbytes + validation_embeddings.nbytes + test_embeddings.nbytes
    ) / (1024**3)
    print(f"Loaded {allocated_gib:.2f} GiB of normalized embeddings into memory.")
    return HomologyData(
        train_essential=train_essential,
        train_nonessential=train_nonessential,
        validation_embeddings=validation_embeddings,
        validation_labels=validation_labels,
        validation_genomes=validation_genomes,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        test_genomes=test_genomes,
    )


def _top_k_similarity_means(
    queries: np.ndarray,
    reference: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Calculate mean cosine similarity over each requested number of neighbors."""
    max_k = max(k_values)
    similarity = queries @ reference.T
    similarity.partition(similarity.shape[1] - max_k, axis=1)
    top_similarity = np.sort(similarity[:, -max_k:], axis=1)[:, ::-1]
    cumulative_similarity = np.cumsum(top_similarity, axis=1)
    return {k: cumulative_similarity[:, k - 1] / k for k in k_values}


def score_by_class_conditional_similarity(
    query_embeddings: np.ndarray,
    train_essential: np.ndarray,
    train_nonessential: np.ndarray,
    k_values: tuple[int, ...],
    progress_description: str,
    query_batch_size: int = QUERY_BATCH_SIZE,
) -> dict[int, ClassConditionalScores]:
    """Score queries by top-k essential minus nonessential mean similarity."""
    if query_batch_size <= 0:
        raise ValueError("Query batch size must be positive.")
    if not k_values or min(k_values) <= 0:
        raise ValueError("At least one positive k value is required.")
    if max(k_values) > min(len(train_essential), len(train_nonessential)):
        raise ValueError("A requested k exceeds the size of a training reference class.")
    k_values = tuple(sorted(set(k_values)))

    n_queries = len(query_embeddings)
    essential_similarity = {k: np.empty(n_queries, dtype=np.float32) for k in k_values}
    nonessential_similarity = {k: np.empty(n_queries, dtype=np.float32) for k in k_values}
    starts = range(0, n_queries, query_batch_size)
    total_batches = math.ceil(n_queries / query_batch_size)

    for start in tqdm(starts, total=total_batches, desc=progress_description, unit="batch"):
        stop = min(start + query_batch_size, n_queries)
        queries = query_embeddings[start:stop]
        essential_batch = _top_k_similarity_means(queries, train_essential, k_values)
        nonessential_batch = _top_k_similarity_means(queries, train_nonessential, k_values)
        for k in k_values:
            essential_similarity[k][start:stop] = essential_batch[k]
            nonessential_similarity[k][start:stop] = nonessential_batch[k]

    return {
        k: ClassConditionalScores(
            margin=essential_similarity[k] - nonessential_similarity[k],
            essential_similarity=essential_similarity[k],
            nonessential_similarity=nonessential_similarity[k],
        )
        for k in k_values
    }


def calculate_metrics_per_genome(
    genome_names: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    essential_similarity: np.ndarray,
    nonessential_similarity: np.ndarray,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Calculate AUROC and AUPRC separately for every genome."""
    predictions = pd.DataFrame(
        {
            "genome_name": genome_names,
            "essential": labels,
            "score": scores,
            "essential_similarity": essential_similarity,
            "nonessential_similarity": nonessential_similarity,
        }
    )
    output = []
    grouped = predictions.groupby("genome_name", sort=True)
    for genome_name, genome in tqdm(
        grouped,
        total=predictions["genome_name"].nunique(),
        desc="Calculating genome metrics",
        unit="genome",
        disable=not show_progress,
    ):
        y_true = genome["essential"].to_numpy(dtype=np.int8)
        y_score = genome["score"].to_numpy(dtype=np.float32)
        if len(np.unique(y_true)) < 2:
            print(f"Skipping {genome_name}: only one class is present.")
            continue
        output.append(
            {
                "genome_name": genome_name,
                "n_proteins": len(genome),
                "n_essential": int(y_true.sum()),
                "essential_fraction": float(y_true.mean()),
                "auroc": float(roc_auc_score(y_true, y_score)),
                "auprc": float(average_precision_score(y_true, y_score)),
                "mean_essential_similarity": float(genome["essential_similarity"].mean()),
                "mean_nonessential_similarity": float(genome["nonessential_similarity"].mean()),
            }
        )

    if not output:
        raise ValueError("No evaluated genomes contained both binary classes.")
    return pd.DataFrame(output)


def tune_k(data: HomologyData, k_values: tuple[int, ...]) -> tuple[int, pd.DataFrame]:
    """Select k by median per-genome validation AUPRC."""
    validation_scores = score_by_class_conditional_similarity(
        query_embeddings=data.validation_embeddings,
        train_essential=data.train_essential,
        train_nonessential=data.train_nonessential,
        k_values=k_values,
        progress_description="Scoring validation proteins",
    )
    output = []
    for k in k_values:
        scores = validation_scores[k]
        per_genome = calculate_metrics_per_genome(
            genome_names=data.validation_genomes,
            labels=data.validation_labels,
            scores=scores.margin,
            essential_similarity=scores.essential_similarity,
            nonessential_similarity=scores.nonessential_similarity,
            show_progress=False,
        )
        summary = summarize_metrics(per_genome).set_index("metric")
        output.append(
            {
                "k": k,
                "median_val_auroc": summary.loc["auroc", "median"],
                "std_val_auroc": summary.loc["auroc", "std"],
                "median_val_auprc": summary.loc["auprc", "median"],
                "std_val_auprc": summary.loc["auprc", "std"],
                "n_validation_genomes": int(summary.loc["auprc", "n_test_genomes"]),
            }
        )

    tuning_results = pd.DataFrame(output)
    best_k = int(tuning_results.sort_values(["median_val_auprc", "k"], ascending=[False, True]).iloc[0]["k"])
    tuning_results["selected"] = tuning_results["k"].eq(best_k)
    return best_k, tuning_results


def summarize_metrics(per_genome: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-genome metrics as median plus sample standard deviation."""
    output = []
    for metric in ("auroc", "auprc"):
        values = per_genome[metric].dropna()
        output.append(
            {
                "metric": metric,
                "median": float(values.median()),
                "std": float(values.std(ddof=1)),
                "n_test_genomes": len(values),
            }
        )
    return pd.DataFrame(output)


def main() -> None:
    """Run the hardcoded CPU-only ESM-C homology baseline."""
    print("Running exact CPU-only ESM-C embedding homology baseline")
    print(f"Input: {INPUT_FILEPATH}")
    layout = inspect_dataset(INPUT_FILEPATH)
    data = load_homology_data(INPUT_FILEPATH, layout)
    best_k, tuning_results = tune_k(data, K_VALUES)
    tuning_results.to_csv(K_TUNING_OUTPUT_FILEPATH, index=False)
    print("\nValidation tuning results:")
    print(tuning_results.to_string(index=False))
    print(f"Selected k={best_k} by median validation-genome AUPRC")

    test_scores = score_by_class_conditional_similarity(
        query_embeddings=data.test_embeddings,
        train_essential=data.train_essential,
        train_nonessential=data.train_nonessential,
        k_values=(best_k,),
        progress_description="Scoring test proteins",
    )[best_k]
    per_genome = calculate_metrics_per_genome(
        genome_names=data.test_genomes,
        labels=data.test_labels,
        scores=test_scores.margin,
        essential_similarity=test_scores.essential_similarity,
        nonessential_similarity=test_scores.nonessential_similarity,
    )
    per_genome.insert(1, "k", best_k)
    summary = summarize_metrics(per_genome)
    summary.insert(1, "k", best_k)

    per_genome.to_csv(PER_GENOME_OUTPUT_FILEPATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_FILEPATH, index=False)

    summary_by_metric = summary.set_index("metric")
    print("\nTest results across genomes:")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Metric & Median & Standard deviation & Median $\pm$ standard deviation \\")
    print(r"\hline")
    for metric in ("auroc", "auprc"):
        median = summary_by_metric.loc[metric, "median"]
        std = summary_by_metric.loc[metric, "std"]
        print(rf"{metric.upper()} & {median:.4f} & {std:.4f} & {median:.4f} $\pm$ {std:.4f} \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(f"Test genomes evaluated: {int(summary_by_metric.loc['auroc', 'n_test_genomes'])}")
    print(f"Saved validation k tuning to: {K_TUNING_OUTPUT_FILEPATH}")
    print(f"Saved per-genome results to: {PER_GENOME_OUTPUT_FILEPATH}")
    print(f"Saved summary to: {SUMMARY_OUTPUT_FILEPATH}")


if __name__ == "__main__":
    main()
