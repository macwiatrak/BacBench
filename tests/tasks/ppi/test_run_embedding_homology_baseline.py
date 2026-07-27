from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bacbench.tasks.ppi import run_embedding_homology_baseline as baseline


def _protein_embeddings() -> list[list[float]]:
    return [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]


def _pair_labels() -> list[list[int]]:
    return [
        [0, 1, 700],
        [1, 0, 700],
        [0, 2, 300],
        [2, 0, 300],
    ]


def test_deduplicate_unordered_pairs_removes_reversed_records():
    labels = np.array(
        [
            [0, 1, 700],
            [1, 0, 700],
            [0, 2, 300],
            [2, 0, 300],
            [0, 5, 900],
        ],
        dtype=np.int64,
    )

    pairs, scores, n_valid = baseline._deduplicate_unordered_pairs(labels, n_proteins=3)

    np.testing.assert_array_equal(pairs, [[0, 1], [0, 2]])
    np.testing.assert_array_equal(scores, [700, 300])
    assert n_valid == 4


def test_pair_embedding_is_symmetric_and_normalized():
    proteins = np.array(_protein_embeddings(), dtype=np.float32)
    baseline._normalize_rows_in_place(proteins, "proteins")

    pairs = baseline._create_pair_embeddings(
        proteins,
        np.array([[0, 1], [1, 0]], dtype=np.int64),
    )

    np.testing.assert_allclose(pairs[0], pairs[1])
    np.testing.assert_allclose(np.linalg.norm(pairs, axis=1), [1.0, 1.0])


def test_prepare_and_build_flattened_contig_rows(tmp_path: Path):
    input_filepath = tmp_path / "ppi_contigs.parquet"
    pd.DataFrame(
        {
            "strain_name": ["train_genome", "train_genome", "val_genome", "test_genome"],
            "split": ["train", "train", "validation", "test"],
            "embeddings": [_protein_embeddings() for _ in range(4)],
            "labels": [_pair_labels() for _ in range(4)],
        }
    ).to_parquet(input_filepath)

    contigs, embedding_dim = baseline.prepare_contigs(input_filepath)
    data = baseline.build_homology_data(contigs, embedding_dim)

    assert embedding_dim == 2
    assert len(contigs) == 4
    assert data.train_interacting.shape == (2, 2)
    assert data.train_noninteracting.shape == (2, 2)
    assert data.validation.embeddings.shape == (2, 2)
    assert data.validation.strain_names.tolist() == ["val_genome", "val_genome"]
    assert data.test.strain_names.tolist() == ["test_genome", "test_genome"]


def test_genome_metrics_aggregate_contigs_by_strain_name():
    per_genome = baseline.calculate_metrics_per_genome(
        strain_names=np.array(["g1", "g2", "g1", "g2"], dtype=object),
        labels=np.array([0, 0, 1, 1], dtype=np.int8),
        scores=np.array([-1.0, -0.5, 1.0, 0.5], dtype=np.float32),
        show_progress=False,
    )
    summary = baseline.summarize_metrics(per_genome).set_index("metric")

    assert per_genome["strain_name"].tolist() == ["g1", "g2"]
    assert per_genome["n_ppi_pairs"].tolist() == [2, 2]
    assert per_genome["auroc"].tolist() == [1.0, 1.0]
    assert per_genome["auprc"].tolist() == [1.0, 1.0]
    assert summary.loc["auroc", "mean"] == 1.0
    assert summary.loc["auroc", "median"] == 1.0
    assert summary.loc["auroc", "std"] == 0.0


def test_tune_k_uses_validation_auprc_and_breaks_ties_toward_smaller_k():
    interacting = np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), 3, axis=0)
    noninteracting = np.repeat(np.array([[0.0, 1.0]], dtype=np.float32), 3, axis=0)
    validation_embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]],
        dtype=np.float32,
    )
    baseline._normalize_rows_in_place(validation_embeddings, "validation")
    validation = baseline.PairSplit(
        embeddings=validation_embeddings,
        labels=np.array([1, 0, 1, 0], dtype=np.int8),
        strain_names=np.array(["g1", "g1", "g2", "g2"], dtype=object),
    )
    data = baseline.HomologyData(
        train_interacting=interacting,
        train_noninteracting=noninteracting,
        validation=validation,
        test=validation,
    )
    reference_index = baseline.build_reference_index(data, device=torch.device("cpu"))

    best_k, tuning_results = baseline.tune_k(data, reference_index, (1, 3))

    assert best_k == 1
    assert tuning_results.loc[tuning_results["selected"], "k"].tolist() == [1]
