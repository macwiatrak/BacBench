import json
import random
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm


def _normalize_labels_array(contig_labels: Any) -> np.ndarray:
    """Convert a contig's labels to a dense (n_pairs, 3) int64 array."""
    labels_array = np.asarray(contig_labels)
    if labels_array.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if labels_array.dtype == np.object_:
        labels_array = np.stack([np.asarray(row, dtype=np.int64) for row in labels_array], axis=0)
    else:
        labels_array = labels_array.astype(np.int64, copy=False)
    return labels_array.reshape(-1, 3)


def _normalize_embeddings_array(contig_embeddings: Any) -> np.ndarray:
    """Convert a contig's embeddings to a dense (n_proteins, dim) float32 array."""
    embeddings_array = np.asarray(contig_embeddings)
    if embeddings_array.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if embeddings_array.dtype == np.object_:
        embeddings_array = np.stack([np.asarray(row, dtype=np.float32) for row in embeddings_array], axis=0)
    else:
        embeddings_array = embeddings_array.astype(np.float32, copy=False)
    return embeddings_array


def _split_rows_from_parquet(
    input_filepath: str,
    train_test_split_filepath: str,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the single PPI parquet incrementally and partition rows by split."""
    with open(train_test_split_filepath) as f:
        split = json.load(f)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    parquet_file = pq.ParquetFile(input_filepath)
    for batch in parquet_file.iter_batches(columns=["strain_name", "labels", "embeddings"], batch_size=batch_size):
        for row in batch.to_pylist():
            split_name = split.get(row["strain_name"])
            if split_name == "train":
                train_rows.append(row)
            elif split_name == "validation":
                val_rows.append(row)
            elif split_name == "test":
                test_rows.append(row)

    return train_rows, val_rows, test_rows


def _infer_hidden_size(*datasets: "PpiDataset") -> int:
    """Infer embedding dimensionality from the first non-empty split."""
    for dataset in datasets:
        if dataset.embeddings.ndim == 2 and dataset.embeddings.shape[1] > 0:
            return int(dataset.embeddings.shape[1])
    raise ValueError(
        "No usable PPI pairs were found in any split, so hidden_size could not be inferred. "
        "Check the input parquet, split file, and filtering thresholds."
    )


class PpiDataset(torch.utils.data.Dataset):
    """Dataset for PPI finetuning.

    The dataset builds one embedding matrix and one label matrix across every genome
    in the split. Each item then samples a single PPI pair from the global label table
    while reusing the global embedding matrix.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        max_n_proteins: int,
        max_n_ppi_pairs: float,
        score_threshold: float | None,
    ):
        self.max_n_proteins = int(max_n_proteins)
        self.max_n_ppi_pairs = int(max_n_ppi_pairs)
        self.score_threshold = score_threshold
        self.rows = rows

        self.embeddings, self.ppi_labels, self.pair_genome_names = self._build_split_matrices()

    def _build_genome_example(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        all_embeddings: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        curr_idx = 0

        for contig_labels, contig_embeddings in zip(row["labels"], row["embeddings"], strict=False):
            labels_array = _normalize_labels_array(contig_labels)
            embeddings_array = _normalize_embeddings_array(contig_embeddings)
            if embeddings_array.size == 0:
                continue

            embeddings_array = embeddings_array[: self.max_n_proteins]
            n_embeddings = embeddings_array.shape[0]
            if n_embeddings == 0:
                continue

            if labels_array.size == 0:
                curr_idx += n_embeddings
                all_embeddings.append(embeddings_array)
                continue

            valid_mask = (labels_array[:, 0] < n_embeddings) & (labels_array[:, 1] < n_embeddings)
            labels_array = labels_array[valid_mask]
            if labels_array.size == 0:
                curr_idx += n_embeddings
                all_embeddings.append(embeddings_array)
                continue

            if self.score_threshold is None:
                labels_array[:, 2] = (labels_array[:, 2] > 0).astype(np.int64)
            else:
                labels_array[:, 2] = (labels_array[:, 2] >= int(self.score_threshold * 1000)).astype(np.int64)

            labels_array[:, :2] += curr_idx
            all_embeddings.append(embeddings_array)
            all_labels.append(labels_array)
            curr_idx += n_embeddings

        if not all_embeddings or not all_labels:
            return None

        embeddings = np.concatenate(all_embeddings, axis=0)
        if embeddings.shape[0] > self.max_n_proteins:
            embeddings = embeddings[: self.max_n_proteins]

        labels = np.concatenate(all_labels, axis=0)
        valid_mask = (labels[:, 0] < embeddings.shape[0]) & (labels[:, 1] < embeddings.shape[0])
        labels = labels[valid_mask]
        if labels.size == 0:
            return None

        if labels.shape[0] > self.max_n_ppi_pairs:
            chosen = np.array(random.sample(range(labels.shape[0]), k=self.max_n_ppi_pairs), dtype=np.int64)
            labels = labels[chosen]

        return embeddings, labels

    def _build_split_matrices(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        all_embeddings: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        pair_genome_names: list[str] = []
        curr_idx = 0

        for row in tqdm(self.rows, desc="Building PPI dataset"):
            genome_example = self._build_genome_example(row)
            if genome_example is None:
                continue
            genome_embeddings, genome_labels = genome_example
            genome_labels = genome_labels.copy()
            genome_labels[:, :2] += curr_idx

            all_embeddings.append(genome_embeddings)
            all_labels.append(genome_labels)
            pair_genome_names.extend([row["strain_name"]] * genome_labels.shape[0])
            curr_idx += genome_embeddings.shape[0]

        if not all_embeddings or not all_labels:
            return (
                torch.empty((0, 0), dtype=torch.float32),
                torch.empty((0, 3), dtype=torch.long),
                [],
            )

        embeddings = torch.as_tensor(np.concatenate(all_embeddings, axis=0), dtype=torch.float32)
        ppi_labels = torch.as_tensor(np.concatenate(all_labels, axis=0), dtype=torch.long)
        return embeddings, ppi_labels, pair_genome_names

    def __len__(self) -> int:
        return int(self.ppi_labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.ppi_labels[idx]
        label = item[2]
        prot1_embedding = self.embeddings[item[0]]
        prot2_embedding = self.embeddings[item[1]]
        return {
            "genome_name": self.pair_genome_names[idx],
            "prot1_embedding": prot1_embedding,
            "prot2_embedding": prot2_embedding,
            "label": label,
        }


def collate_ppi(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, Any]:
    """Collate PPI examples."""
    prot1_embeddings = torch.stack([item["prot1_embedding"] for item in batch])
    prot2_embeddings = torch.stack([item["prot2_embedding"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    genome_names = [item["genome_name"] for item in batch]
    return {
        "genome_names": genome_names,
        "prot1_embeddings": prot1_embeddings,
        "prot2_embeddings": prot2_embeddings,
        "labels": labels,
    }


def get_datasets_ppi(
    input_filepath: str,
    train_test_split_filepath: str,
    max_n_proteins: int,
    max_n_ppi_pairs: float,
    score_threshold: float | None,
) -> tuple[PpiDataset, PpiDataset, PpiDataset, int]:
    """Build train/val/test datasets for the single-parquet PPI format."""
    train_rows, val_rows, test_rows = _split_rows_from_parquet(input_filepath, train_test_split_filepath)

    train_ds = PpiDataset(
        rows=train_rows,
        max_n_proteins=max_n_proteins,
        max_n_ppi_pairs=max_n_ppi_pairs,
        score_threshold=score_threshold,
    )
    val_ds = PpiDataset(
        rows=val_rows,
        max_n_proteins=max_n_proteins,
        max_n_ppi_pairs=max_n_ppi_pairs,
        score_threshold=score_threshold,
    )
    test_ds = PpiDataset(
        rows=test_rows,
        max_n_proteins=max_n_proteins,
        max_n_ppi_pairs=max_n_ppi_pairs,
        score_threshold=score_threshold,
    )
    hidden_size = _infer_hidden_size(train_ds, val_ds, test_ds)
    return train_ds, val_ds, test_ds, hidden_size


def get_dataloaders_ppi(
    input_filepath: str,
    train_test_split_filepath: str,
    max_n_proteins: int,
    max_n_ppi_pairs: float,
    score_threshold: float | None,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader | None, int]:
    """Get train/val/test dataloaders for the single-parquet PPI format.

    The parquet input is expected to match `run_unsupervised_eval.py` and contain
    `strain_name`, `labels`, and `embeddings` columns. The split JSON must map
    strain names to `train`, `validation`, or `test`.
    """
    train_ds, val_ds, test_ds, hidden_size = get_datasets_ppi(
        input_filepath=input_filepath,
        train_test_split_filepath=train_test_split_filepath,
        max_n_proteins=max_n_proteins,
        max_n_ppi_pairs=max_n_ppi_pairs,
        score_threshold=score_threshold,
    )

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ppi,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ppi,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ppi,
    )
    return train_dl, val_dl, test_dl, hidden_size
