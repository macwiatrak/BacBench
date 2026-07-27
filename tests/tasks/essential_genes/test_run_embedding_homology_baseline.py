import importlib

import numpy as np

baseline = importlib.import_module("bacbench.tasks.essential_genes.run_embedding_homology_baseline")


def test_class_conditional_similarity_margin():
    train_essential = np.array([[1.0, 0.0]], dtype=np.float32)
    train_nonessential = np.array([[0.0, 1.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    baseline._normalize_rows_in_place(queries, "queries")

    result = baseline.score_by_class_conditional_similarity(
        query_embeddings=queries,
        train_essential=train_essential,
        train_nonessential=train_nonessential,
        k_values=(1,),
        progress_description="Testing",
        query_batch_size=2,
    )[1]

    np.testing.assert_allclose(result.essential_similarity, [1.0, 0.0, 2**-0.5])
    np.testing.assert_allclose(result.nonessential_similarity, [0.0, 1.0, 2**-0.5])
    np.testing.assert_allclose(result.margin, [1.0, -1.0, 0.0])


def test_top_k_similarity_means():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    reference = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = baseline._top_k_similarity_means(queries, reference, (1, 3))

    np.testing.assert_allclose(result[1], [1.0])
    np.testing.assert_allclose(result[3], [0.6])


def test_tune_k_prefers_smallest_k_when_validation_auprc_ties():
    train_essential = np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), 3, axis=0)
    train_nonessential = np.repeat(np.array([[0.0, 1.0]], dtype=np.float32), 3, axis=0)
    validation_embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, 0.8]],
        dtype=np.float32,
    )
    baseline._normalize_rows_in_place(validation_embeddings, "validation")
    data = baseline.HomologyData(
        train_essential=train_essential,
        train_nonessential=train_nonessential,
        validation_embeddings=validation_embeddings,
        validation_labels=np.array([1, 0, 1, 0], dtype=np.int8),
        validation_genomes=np.array(["g1", "g1", "g2", "g2"], dtype=object),
        test_embeddings=np.empty((0, 2), dtype=np.float32),
        test_labels=np.empty(0, dtype=np.int8),
        test_genomes=np.empty(0, dtype=object),
    )

    best_k, tuning_results = baseline.tune_k(data, (1, 3))

    assert best_k == 1
    assert tuning_results.loc[tuning_results["selected"], "k"].tolist() == [1]


def test_calculate_and_summarize_per_genome_metrics():
    genome_names = np.array(["g1", "g1", "g2", "g2"], dtype=object)
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    scores = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
    essential_similarity = np.array([0.0, 1.0, 0.25, 0.75], dtype=np.float32)
    nonessential_similarity = np.array([1.0, 0.0, 0.75, 0.25], dtype=np.float32)

    per_genome = baseline.calculate_metrics_per_genome(
        genome_names,
        labels,
        scores,
        essential_similarity,
        nonessential_similarity,
    )
    summary = baseline.summarize_metrics(per_genome).set_index("metric")

    assert per_genome["auroc"].tolist() == [1.0, 1.0]
    assert per_genome["auprc"].tolist() == [1.0, 1.0]
    assert summary.loc["auroc", "median"] == 1.0
    assert summary.loc["auroc", "std"] == 0.0
    assert summary.loc["auprc", "median"] == 1.0
    assert summary.loc["auprc", "std"] == 0.0
