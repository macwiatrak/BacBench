import itertools
import os
import random
from functools import partial
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset


def process_triples(
    triples: list[tuple[int, int, int]],
    score_threshold: float | None,
    max_n_proteins: int,
    max_n_ppi_pairs: float = 2 * 1e6,
) -> torch.Tensor:
    """Convert PPI triples to an interaction matrix which we can use as labels."""
    triples = torch.tensor(np.stack(triples, axis=1), dtype=torch.long)
    # binarize the scores
    # STRING DB scores are divided by 1000, i.e. 0.476 means 476, so we need to multiply by 1000
    score_threshold = int(score_threshold * 1000) if score_threshold is not None else 0
    triples[2] = (triples[2] >= score_threshold).long()

    # limit to max_n_ppi_pairs
    if triples.shape[1] > max_n_ppi_pairs:
        ppi_pairs_indices = random.sample(range(triples.shape[1]), k=int(max_n_ppi_pairs))
        triples = triples[:, ppi_pairs_indices]

    if triples[:2, :].max() < max_n_proteins:
        # print("triples normal", triples.shape)
        return triples
    # limit to max_n_proteins
    mask = (triples[0] < max_n_proteins) & (triples[1] < max_n_proteins)
    triples = triples[:, mask]
    # print("triples masked", triples.shape)
    return triples


def transform_sample(
    max_n_proteins: int,
    max_n_ppi_pairs: int,
    score_threshold: float | None,  # if score threshold is None, then do not set threshold
    embeddings_col: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Transform the sample including the labels."""
    if isinstance(item[embeddings_col][0], list):
        # if the embeddings are a list of lists, flatten them
        item[embeddings_col] = list(itertools.chain(*item[embeddings_col]))
    item["ppi_labels"] = process_triples(
        triples=item["triples_combined_score"],
        score_threshold=score_threshold,
        max_n_proteins=min(len(item[embeddings_col]), max_n_proteins),  # account for CLS, SEP, and END tokens
        max_n_ppi_pairs=max_n_ppi_pairs,
    )
    del item["triples_combined_score"]
    item["embeddings"] = torch.stack([torch.tensor(i) for i in item[embeddings_col]])
    return item


def collate_ppi(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate the batch."""
    # collate the protein embeddings
    protein_embeddings = torch.stack([item["embeddings"] for item in batch])
    # collate the labels
    ppi_labels = torch.stack([item["ppi_labels"] for item in batch])
    return {
        "genome_name": [item.get("genome_name", None) for item in batch],
        "protein_embeddings": protein_embeddings,
        "ppi_labels": ppi_labels,
    }


def get_datasets_ppi(
    input_dir: str,
    max_n_proteins: int,
    max_n_ppi_pairs: float,
    # a list of PPI scores from STRING DB to include, e.g. ["combined_score"]
    # for finetuning use only one column, for unsupervised evaluation potentially use multiple
    score_threshold: float | None,
    test: bool = False,
    embeddings_col: str = "embeddings",
) -> tuple[Dataset, Dataset, Dataset | None]:
    """Get the datasets."""
    # get the files for train, val and test
    train_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith("train")]
    val_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith("val")]
    test_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith("test")]
    data_files = {"train": train_files, "validation": val_files, "test": test_files}
    cols_to_use = ["genome_name", embeddings_col, "triples_combined_score"]

    transform_fn = partial(transform_sample, max_n_proteins, max_n_ppi_pairs, score_threshold, embeddings_col)
    train_dataset = (
        load_dataset("parquet", data_files=data_files, split="train", streaming=True)
        .select_columns(cols_to_use)
        .map(transform_fn, batched=False, with_indices=False)
    )
    val_dataset = (
        load_dataset("parquet", data_files=data_files, split="validation", streaming=True)
        .select_columns(cols_to_use)
        .map(transform_fn, batched=False, with_indices=False)
    )

    if not test:
        return train_dataset, val_dataset, None

    test_dataset = (
        load_dataset("parquet", data_files={"test": test_files}, split="test", streaming=True)
        .select_columns(cols_to_use)
        .map(transform_fn, batched=False, with_indices=False)
    )
    return train_dataset, val_dataset, test_dataset
