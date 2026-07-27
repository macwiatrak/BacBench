from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

INPUT_FILEPATH = Path("/Users/maciejwiatrak/Downloads/esmc_ppi.parquet")
TRAIN_TEST_SPLIT_FILEPATH = Path("/projects/public/u6fp/benchmarks/tasks/ppi/eval/strain_split.json")
PER_GENOME_OUTPUT_FILEPATH = Path("/Users/maciejwiatrak/Downloads/esmc_ppi_homology_baseline_per_genome.csv")
SUMMARY_OUTPUT_FILEPATH = Path("/Users/maciejwiatrak/Downloads/esmc_ppi_homology_baseline_summary.csv")
K_TUNING_OUTPUT_FILEPATH = Path("/Users/maciejwiatrak/Downloads/esmc_ppi_homology_baseline_k_tuning.csv")

K_VALUES = (1, 3, 5, 10)
SCORE_THRESHOLD = 0.6
MAX_N_PROTEINS = 6000
MAX_N_PPI_PAIRS = 2_000_000
PAIR_EMBEDDING_BATCH_SIZE = 2048
CPU_QUERY_BATCH_SIZE = 64
CUDA_QUERY_BATCH_SIZE = 512
REQUIRED_COLUMNS = {"strain_name", "split", "labels", "embeddings"}
REQUIRED_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class PreparedContig:
    """Validated data for one contig row."""

    strain_name: str
    split: str
    protein_embeddings: np.ndarray
    pair_indices: np.ndarray
    pair_labels: np.ndarray
    n_raw_pairs: int
    n_valid_pairs: int


@dataclass(frozen=True)
class PairSplit:
    """Pair embeddings and metadata for a query split."""

    embeddings: np.ndarray
    labels: np.ndarray
    strain_names: np.ndarray


@dataclass(frozen=True)
class HomologyData:
    """Class-conditional train references and validation/test queries."""

    train_interacting: np.ndarray
    train_noninteracting: np.ndarray
    validation: PairSplit
    test: PairSplit


@dataclass(frozen=True)
class ClassConditionalScores:
    """PPI score and its class-conditional similarity components."""

    margin: np.ndarray
    interacting_similarity: np.ndarray
    noninteracting_similarity: np.ndarray


@dataclass(frozen=True)
class ReferenceIndex:
    """Class-conditional reference matrices resident on the search device."""

    interacting: torch.Tensor
    noninteracting: torch.Tensor
    device: torch.device


def _normalize_embeddings_array(values: Any) -> np.ndarray:
    """Convert one contig's protein embeddings to a dense float32 matrix."""
    embeddings = np.asarray(values)
    if embeddings.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if embeddings.dtype == np.object_:
        embeddings = np.stack([np.asarray(row, dtype=np.float32) for row in embeddings], axis=0)
    else:
        embeddings = embeddings.astype(np.float32, copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D protein embedding matrix, observed shape {embeddings.shape}.")
    return embeddings


def _normalize_labels_array(values: Any) -> np.ndarray:
    """Convert one contig's labels to a dense (n_pairs, 3) int64 matrix."""
    labels = np.asarray(values)
    if labels.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if labels.dtype == np.object_:
        labels = np.stack([np.asarray(row, dtype=np.int64) for row in labels], axis=0)
    else:
        labels = labels.astype(np.int64, copy=False)
    if labels.size % 3:
        raise ValueError(f"PPI labels cannot be reshaped to (n_pairs, 3); observed shape {labels.shape}.")
    return labels.reshape(-1, 3)


def _normalize_rows_in_place(matrix: np.ndarray, name: str) -> None:
    """L2-normalize a matrix in place for cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError(f"{name} contains non-finite embeddings.")
    if np.any(norms == 0):
        raise ValueError(f"{name} contains zero-norm embeddings.")
    matrix /= norms[:, None]


def _deduplicate_unordered_pairs(
    labels: np.ndarray,
    n_proteins: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate indices and collapse duplicated (i, j)/(j, i) records."""
    if labels.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), 0

    valid_mask = (labels[:, 0] >= 0) & (labels[:, 1] >= 0) & (labels[:, 0] < n_proteins) & (labels[:, 1] < n_proteins)
    valid_labels = labels[valid_mask]
    if valid_labels.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), 0

    unordered_pairs = np.sort(valid_labels[:, :2], axis=1)
    unique_pairs, first_indices, inverse = np.unique(
        unordered_pairs,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    scores = valid_labels[:, 2]
    min_scores = np.full(len(unique_pairs), np.iinfo(np.int64).max, dtype=np.int64)
    max_scores = np.full(len(unique_pairs), np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(min_scores, inverse, scores)
    np.maximum.at(max_scores, inverse, scores)
    if np.any(min_scores != max_scores):
        raise ValueError("An undirected protein pair has conflicting STRING scores.")

    return unique_pairs, scores[first_indices], int(valid_mask.sum())


def prepare_contigs(input_filepath: Path, train_test_split_filepath: str) -> tuple[list[PreparedContig], int]:
    """Read and validate flattened contig rows from the PPI parquet."""
    rows = pd.read_parquet(input_filepath)

    with open(train_test_split_filepath) as f:
        split_dict = json.load(f)
    rows["split"] = rows["strain_name"].map(split_dict)

    rows = rows[["strain_name", "embeddings", "split", "labels"]].explode(["embeddings", "labels"])
    missing_columns = REQUIRED_COLUMNS.difference(rows.columns)
    if missing_columns:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing_columns)}")
    rows = rows[["strain_name", "split", "labels", "embeddings"]]
    if rows[["strain_name", "split"]].isna().any().any():
        raise ValueError("Strain names and split values must not be missing.")

    observed_splits = set(rows["split"].unique())
    if observed_splits != REQUIRED_SPLITS:
        raise ValueError(f"Expected exactly train, validation, and test splits; observed {sorted(observed_splits)}.")
    leaking_strains = rows.groupby("strain_name")["split"].nunique()
    leaking_strains = leaking_strains[leaking_strains > 1]
    if not leaking_strains.empty:
        examples = leaking_strains.index.astype(str).tolist()[:5]
        raise ValueError(f"Strains occur in more than one split, for example: {examples}")

    prepared: list[PreparedContig] = []
    embedding_dim: int | None = None
    for row_number, row in tqdm(
        rows.iterrows(),
        total=len(rows),
        desc="Preparing contigs",
        unit="contig",
    ):
        embeddings = _normalize_embeddings_array(row["embeddings"])
        embeddings = embeddings[:MAX_N_PROTEINS].copy()
        if embeddings.size == 0:
            continue
        if embedding_dim is None:
            embedding_dim = int(embeddings.shape[1])
        elif embeddings.shape[1] != embedding_dim:
            raise ValueError(
                f"Contig row {row_number} has embedding dimension {embeddings.shape[1]}; expected {embedding_dim}."
            )
        _normalize_rows_in_place(embeddings, f"Protein embeddings in contig row {row_number}")

        raw_labels = _normalize_labels_array(row["labels"])
        raw_labels = raw_labels[:MAX_N_PPI_PAIRS]
        pair_indices, string_scores, n_valid_pairs = _deduplicate_unordered_pairs(
            raw_labels,
            len(embeddings),
        )
        if not len(pair_indices):
            continue
        pair_labels = (string_scores.astype(np.float32) / 1000.0 >= SCORE_THRESHOLD).astype(np.int8)
        prepared.append(
            PreparedContig(
                strain_name=str(row["strain_name"]),
                split=str(row["split"]),
                protein_embeddings=embeddings,
                pair_indices=pair_indices,
                pair_labels=pair_labels,
                n_raw_pairs=len(raw_labels),
                n_valid_pairs=n_valid_pairs,
            )
        )

    if embedding_dim is None or not prepared:
        raise ValueError("No usable protein embeddings and PPI pairs were found.")

    summary_rows = []
    for split in ("train", "validation", "test"):
        split_contigs = [contig for contig in prepared if contig.split == split]
        if not split_contigs:
            raise ValueError(f"No usable PPI pairs were found in the {split} split.")
        labels = np.concatenate([contig.pair_labels for contig in split_contigs])
        summary_rows.append(
            {
                "split": split,
                "contigs": len(split_contigs),
                "strains": len({contig.strain_name for contig in split_contigs}),
                "proteins": sum(len(contig.protein_embeddings) for contig in split_contigs),
                "raw_pairs": sum(contig.n_raw_pairs for contig in split_contigs),
                "valid_pairs": sum(contig.n_valid_pairs for contig in split_contigs),
                "unique_pairs": len(labels),
                "positive_pairs": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("split")
    if summary.loc["train", "positive_pairs"] in (0, summary.loc["train", "unique_pairs"]):
        raise ValueError("The training split must contain both binary PPI classes.")
    print("Dataset summary after removing reversed-pair duplicates:")
    print(summary.to_string())
    print(f"Protein and pair embedding dimension: {embedding_dim}")
    return prepared, embedding_dim


def _create_pair_embeddings(
    protein_embeddings: np.ndarray,
    pair_indices: np.ndarray,
) -> np.ndarray:
    """Create symmetric pair embeddings by averaging normalized partners."""
    pair_embeddings = protein_embeddings[pair_indices[:, 0]] + protein_embeddings[pair_indices[:, 1]]
    _normalize_rows_in_place(pair_embeddings, "Pair embeddings")
    return pair_embeddings


def build_homology_data(
    contigs: list[PreparedContig],
    embedding_dim: int,
) -> HomologyData:
    """Materialize class references and query pairs in memory-bounded batches."""
    train_positive_count = sum(int(contig.pair_labels.sum()) for contig in contigs if contig.split == "train")
    train_total_count = sum(len(contig.pair_labels) for contig in contigs if contig.split == "train")
    train_negative_count = train_total_count - train_positive_count
    validation_count = sum(len(contig.pair_labels) for contig in contigs if contig.split == "validation")
    test_count = sum(len(contig.pair_labels) for contig in contigs if contig.split == "test")

    train_interacting = np.empty((train_positive_count, embedding_dim), dtype=np.float32)
    train_noninteracting = np.empty((train_negative_count, embedding_dim), dtype=np.float32)
    validation_embeddings = np.empty((validation_count, embedding_dim), dtype=np.float32)
    validation_labels = np.empty(validation_count, dtype=np.int8)
    validation_strains = np.empty(validation_count, dtype=object)
    test_embeddings = np.empty((test_count, embedding_dim), dtype=np.float32)
    test_labels = np.empty(test_count, dtype=np.int8)
    test_strains = np.empty(test_count, dtype=object)

    cursors = {"train_positive": 0, "train_negative": 0, "validation": 0, "test": 0}
    total_pairs = sum(len(contig.pair_labels) for contig in contigs)
    with tqdm(total=total_pairs, desc="Creating pair embeddings", unit="pair") as progress:
        for contig in contigs:
            for start in range(0, len(contig.pair_labels), PAIR_EMBEDDING_BATCH_SIZE):
                stop = min(start + PAIR_EMBEDDING_BATCH_SIZE, len(contig.pair_labels))
                pair_labels = contig.pair_labels[start:stop]
                pair_embeddings = _create_pair_embeddings(
                    contig.protein_embeddings,
                    contig.pair_indices[start:stop],
                )

                if contig.split == "train":
                    positive_mask = pair_labels == 1
                    n_positive = int(positive_mask.sum())
                    positive_cursor = cursors["train_positive"]
                    train_interacting[positive_cursor : positive_cursor + n_positive] = pair_embeddings[positive_mask]
                    cursors["train_positive"] += n_positive

                    n_negative = len(pair_labels) - n_positive
                    negative_cursor = cursors["train_negative"]
                    train_noninteracting[negative_cursor : negative_cursor + n_negative] = pair_embeddings[
                        ~positive_mask
                    ]
                    cursors["train_negative"] += n_negative
                else:
                    cursor = cursors[contig.split]
                    next_cursor = cursor + len(pair_labels)
                    if contig.split == "validation":
                        validation_embeddings[cursor:next_cursor] = pair_embeddings
                        validation_labels[cursor:next_cursor] = pair_labels
                        validation_strains[cursor:next_cursor] = contig.strain_name
                    else:
                        test_embeddings[cursor:next_cursor] = pair_embeddings
                        test_labels[cursor:next_cursor] = pair_labels
                        test_strains[cursor:next_cursor] = contig.strain_name
                    cursors[contig.split] = next_cursor
                progress.update(len(pair_labels))

    expected_cursors = {
        "train_positive": train_positive_count,
        "train_negative": train_negative_count,
        "validation": validation_count,
        "test": test_count,
    }
    if cursors != expected_cursors:
        raise RuntimeError(f"Loaded pair counts {cursors} do not match expected counts {expected_cursors}.")

    allocated_gib = (
        train_interacting.nbytes + train_noninteracting.nbytes + validation_embeddings.nbytes + test_embeddings.nbytes
    ) / (1024**3)
    print(f"Created {allocated_gib:.2f} GiB of normalized pair embeddings.")
    return HomologyData(
        train_interacting=train_interacting,
        train_noninteracting=train_noninteracting,
        validation=PairSplit(validation_embeddings, validation_labels, validation_strains),
        test=PairSplit(test_embeddings, test_labels, test_strains),
    )


def _top_k_similarity_means(
    queries: torch.Tensor,
    reference: torch.Tensor,
    k_values: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Calculate mean cosine similarity over each requested number of neighbors."""
    max_k = max(k_values)
    similarity = queries @ reference.T
    top_similarity = torch.topk(similarity, k=max_k, dim=1, largest=True, sorted=True).values
    cumulative_similarity = torch.cumsum(top_similarity, dim=1)
    return {k: cumulative_similarity[:, k - 1] / k for k in k_values}


def build_reference_index(
    data: HomologyData,
    device: torch.device | None = None,
) -> ReferenceIndex:
    """Transfer the train reference database to CUDA once, with a CPU fallback."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU for exact neighbor search: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is unavailable; using the CPU fallback for exact neighbor search.")
    return ReferenceIndex(
        interacting=torch.as_tensor(data.train_interacting, dtype=torch.float32, device=device),
        noninteracting=torch.as_tensor(data.train_noninteracting, dtype=torch.float32, device=device),
        device=device,
    )


def score_by_class_conditional_similarity(
    query_embeddings: np.ndarray,
    reference_index: ReferenceIndex,
    k_values: tuple[int, ...],
    progress_description: str,
    query_batch_size: int | None = None,
) -> dict[int, ClassConditionalScores]:
    """Score query pairs by interacting minus non-interacting top-k similarity."""
    if query_batch_size is None:
        query_batch_size = CUDA_QUERY_BATCH_SIZE if reference_index.device.type == "cuda" else CPU_QUERY_BATCH_SIZE
    if query_batch_size <= 0:
        raise ValueError("Query batch size must be positive.")
    if not k_values or min(k_values) <= 0:
        raise ValueError("At least one positive k value is required.")
    if max(k_values) > min(len(reference_index.interacting), len(reference_index.noninteracting)):
        raise ValueError("A requested k exceeds the size of a training reference class.")
    k_values = tuple(sorted(set(k_values)))

    n_queries = len(query_embeddings)
    interacting_similarity = {k: np.empty(n_queries, dtype=np.float32) for k in k_values}
    noninteracting_similarity = {k: np.empty(n_queries, dtype=np.float32) for k in k_values}
    starts = range(0, n_queries, query_batch_size)
    total_batches = math.ceil(n_queries / query_batch_size)
    with torch.inference_mode():
        for start in tqdm(starts, total=total_batches, desc=progress_description, unit="batch"):
            stop = min(start + query_batch_size, n_queries)
            queries = torch.as_tensor(
                np.ascontiguousarray(query_embeddings[start:stop]),
                dtype=torch.float32,
                device=reference_index.device,
            )
            interacting_batch = _top_k_similarity_means(
                queries,
                reference_index.interacting,
                k_values,
            )
            noninteracting_batch = _top_k_similarity_means(
                queries,
                reference_index.noninteracting,
                k_values,
            )
            for k in k_values:
                interacting_similarity[k][start:stop] = interacting_batch[k].cpu().numpy()
                noninteracting_similarity[k][start:stop] = noninteracting_batch[k].cpu().numpy()

    return {
        k: ClassConditionalScores(
            margin=interacting_similarity[k] - noninteracting_similarity[k],
            interacting_similarity=interacting_similarity[k],
            noninteracting_similarity=noninteracting_similarity[k],
        )
        for k in k_values
    }


def calculate_metrics_per_genome(
    strain_names: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Aggregate all contigs and calculate metrics separately for each strain."""
    predictions = pd.DataFrame(
        {
            "strain_name": strain_names,
            "label": labels,
            "score": scores,
        }
    )
    output = []
    grouped = predictions.groupby("strain_name", sort=True)
    for strain_name, genome in tqdm(
        grouped,
        total=predictions["strain_name"].nunique(),
        desc="Calculating genome metrics",
        unit="genome",
        disable=not show_progress,
    ):
        y_true = genome["label"].to_numpy(dtype=np.int8)
        y_score = genome["score"].to_numpy(dtype=np.float32)
        if len(np.unique(y_true)) < 2:
            print(f"Skipping {strain_name}: only one PPI class is present.")
            continue
        output.append(
            {
                "strain_name": strain_name,
                "n_ppi_pairs": len(genome),
                "n_positive_ppi_pairs": int(y_true.sum()),
                "positive_fraction": float(y_true.mean()),
                "auroc": float(roc_auc_score(y_true, y_score)),
                "auprc": float(average_precision_score(y_true, y_score)),
            }
        )
    if not output:
        raise ValueError("No evaluated genomes contained both binary PPI classes.")
    return pd.DataFrame(output)


def summarize_metrics(per_genome: pd.DataFrame) -> pd.DataFrame:
    """Report mean, median, and sample standard deviation across genomes."""
    output = []
    for metric in ("auroc", "auprc"):
        values = per_genome[metric].dropna()
        output.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)),
                "n_genomes": len(values),
            }
        )
    return pd.DataFrame(output)


def tune_k(
    data: HomologyData,
    reference_index: ReferenceIndex,
    k_values: tuple[int, ...],
) -> tuple[int, pd.DataFrame]:
    """Select k using median per-genome validation AUPRC."""
    validation_scores = score_by_class_conditional_similarity(
        query_embeddings=data.validation.embeddings,
        reference_index=reference_index,
        k_values=k_values,
        progress_description="Scoring validation PPI pairs",
    )
    output = []
    for k in k_values:
        per_genome = calculate_metrics_per_genome(
            strain_names=data.validation.strain_names,
            labels=data.validation.labels,
            scores=validation_scores[k].margin,
            show_progress=False,
        )
        summary = summarize_metrics(per_genome).set_index("metric")
        output.append(
            {
                "k": k,
                "mean_val_auroc": summary.loc["auroc", "mean"],
                "median_val_auroc": summary.loc["auroc", "median"],
                "std_val_auroc": summary.loc["auroc", "std"],
                "mean_val_auprc": summary.loc["auprc", "mean"],
                "median_val_auprc": summary.loc["auprc", "median"],
                "std_val_auprc": summary.loc["auprc", "std"],
                "n_validation_genomes": int(summary.loc["auprc", "n_genomes"]),
            }
        )

    tuning_results = pd.DataFrame(output)
    best_k = int(
        tuning_results.sort_values(
            ["median_val_auprc", "k"],
            ascending=[False, True],
        ).iloc[0]["k"]
    )
    tuning_results["selected"] = tuning_results["k"].eq(best_k)
    return best_k, tuning_results


def print_test_summary(summary: pd.DataFrame) -> None:
    """Print copy-ready aggregate metrics."""
    summary_by_metric = summary.set_index("metric")
    print("\nTest results across genomes:")
    for metric in ("auroc", "auprc"):
        mean = summary_by_metric.loc[metric, "mean"]
        median = summary_by_metric.loc[metric, "median"]
        std = summary_by_metric.loc[metric, "std"]
        print(f"{metric.upper()}: mean {mean:.4f}, median {median:.4f}, std {std:.4f}")
    print(f"Test genomes evaluated: {int(summary_by_metric.loc['auroc', 'n_genomes'])}")


def main() -> None:
    """Run the hardcoded GPU-accelerated PPI embedding homology baseline."""
    print("Running exact ESM-C PPI embedding homology baseline")
    print(f"Input: {INPUT_FILEPATH}")
    contigs, embedding_dim = prepare_contigs(INPUT_FILEPATH, TRAIN_TEST_SPLIT_FILEPATH)
    data = build_homology_data(contigs, embedding_dim)
    reference_index = build_reference_index(data)

    best_k, tuning_results = tune_k(data, reference_index, K_VALUES)
    tuning_results.to_csv(K_TUNING_OUTPUT_FILEPATH, index=False)
    print("\nValidation tuning results:")
    print(tuning_results.to_string(index=False))
    print(f"Selected k={best_k} by median validation-genome AUPRC")

    test_scores = score_by_class_conditional_similarity(
        query_embeddings=data.test.embeddings,
        reference_index=reference_index,
        k_values=(best_k,),
        progress_description="Scoring test PPI pairs",
    )[best_k]
    per_genome = calculate_metrics_per_genome(
        strain_names=data.test.strain_names,
        labels=data.test.labels,
        scores=test_scores.margin,
    )
    per_genome.insert(1, "k", best_k)
    summary = summarize_metrics(per_genome)
    summary.insert(1, "k", best_k)
    per_genome.to_csv(PER_GENOME_OUTPUT_FILEPATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_FILEPATH, index=False)

    print_test_summary(summary)
    print(f"Saved validation k tuning to: {K_TUNING_OUTPUT_FILEPATH}")
    print(f"Saved per-genome results to: {PER_GENOME_OUTPUT_FILEPATH}")
    print(f"Saved summary to: {SUMMARY_OUTPUT_FILEPATH}")


if __name__ == "__main__":
    main()
