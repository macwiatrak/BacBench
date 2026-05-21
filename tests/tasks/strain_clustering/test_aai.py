import numpy as np
import pandas as pd
import pytest
from bacbench.tasks.strain_clustering.aai import (
    add_protein_id_columns,
    build_aai_distance_matrices,
    cluster_distance_matrix,
    compute_leiden_aai_metrics,
    compute_pairwise_aai,
    evaluate_species_clustering,
    filter_mmseqs_hits,
    flatten_protein_sequences,
    make_protein_id,
    parse_protein_id,
    prepare_genome_dataframe,
    prepare_genomes_from_input,
    rank_leiden_metrics,
    run_mmseqs_all_vs_all,
    select_final_clusters,
    write_protein_fasta,
)


def test_flatten_protein_sequences_accepts_flat_and_contig_grouped_inputs():
    assert flatten_protein_sequences(["AAA", "BBB"]) == ["AAA", "BBB"]
    assert flatten_protein_sequences(np.array([np.array(["AAA", ""]), np.array(["BBB"])], dtype=object)) == [
        "AAA",
        "BBB",
    ]


def test_prepare_genome_dataframe_accepts_sample_protein_sequence_schema():
    df = pd.DataFrame(
        {
            "genome_name": ["genome_a", "genome_b"],
            "species": ["species_a", "species_b"],
            "protein_sequence": [
                [["AAA", "BBB"], ["CCC"]],
                [["DDD"]],
            ],
        }
    )

    prepared = prepare_genome_dataframe(df, genome_col="genome_name", min_species_count=1)

    assert prepared["genome_id"].tolist() == ["genome_a", "genome_b"]
    assert prepared["protein_sequence"].tolist() == [["AAA", "BBB", "CCC"], ["DDD"]]
    assert prepared["n_proteins"].tolist() == [3, 1]


def test_prepare_genomes_from_input_streams_parquet_and_writes_fasta(tmp_path):
    input_path = tmp_path / "genomes.parquet"
    df = pd.DataFrame(
        {
            "genome_name": ["genome_a", "genome_b"],
            "species": ["species_a", "species_b"],
            "protein_sequence": [
                [["AAA", "BBB"], ["CCC"]],
                [["DDD"]],
            ],
        }
    )
    df.to_parquet(input_path)

    prepared = prepare_genomes_from_input(
        input_df_filepath=input_path,
        fasta_path=tmp_path / "proteins.faa",
        protein_index_path=tmp_path / "protein_index.csv",
        min_species_count=1,
        input_batch_size=1,
        show_progress=False,
    )

    assert prepared["genome_id"].tolist() == ["genome_a", "genome_b"]
    assert prepared["n_proteins"].tolist() == [3, 1]
    assert (tmp_path / "proteins.faa").read_text() == ">g0|p0\nAAA\n>g0|p1\nBBB\n>g0|p2\nCCC\n>g1|p0\nDDD\n"
    assert pd.read_csv(tmp_path / "protein_index.csv")["protein_id"].tolist() == ["g0|p0", "g0|p1", "g0|p2", "g1|p0"]


def test_protein_ids_round_trip_and_fasta_mapping_is_written(tmp_path):
    df = pd.DataFrame(
        {
            "genome_id": ["genome_a"],
            "species": ["species_a"],
            "protein_sequence": [[["AAA"], ["BBB"]]],
        }
    )
    prepared = prepare_genome_dataframe(df, min_species_count=1)
    protein_index = write_protein_fasta(prepared, tmp_path / "proteins.faa")

    assert parse_protein_id(make_protein_id(3, 7)) == (3, 7)
    assert protein_index["protein_id"].tolist() == ["g0|p0", "g0|p1"]
    assert (tmp_path / "proteins.faa").read_text() == ">g0|p0\nAAA\n>g0|p1\nBBB\n"


def test_run_mmseqs_all_vs_all_sets_split_memory_limit(tmp_path, monkeypatch):
    captured_commands = []

    def fake_run(command, check):
        captured_commands.append(command)
        assert check is True

    monkeypatch.setattr("bacbench.tasks.strain_clustering.aai.subprocess.run", fake_run)

    output_path = run_mmseqs_all_vs_all(
        fasta_path=tmp_path / "proteins.faa",
        output_tsv_path=tmp_path / "mmseqs_hits.tsv",
        tmp_dir=tmp_path / "mmseqs_tmp",
        threads=8,
    )

    assert output_path == tmp_path / "mmseqs_hits.tsv"
    assert captured_commands[0][-4:] == ["--threads", "8", "--split-memory-limit", "110G"]


def test_pairwise_aai_uses_reciprocal_best_hits_and_flags_missing_pairs():
    df = pd.DataFrame(
        {
            "genome_id": ["genome_a", "genome_b", "genome_c"],
            "species": ["species_1", "species_1", "species_2"],
            "protein_sequence": [["AAA", "BBB"], ["AAC", "CCC"], ["XXX", "YYY"]],
        }
    )
    prepared = prepare_genome_dataframe(df, min_species_count=1)
    hits = pd.DataFrame(
        [
            {
                "query": "g0|p0",
                "target": "g1|p0",
                "pident": 95.0,
                "alnlen": 3,
                "qlen": 3,
                "tlen": 3,
                "evalue": 1e-20,
                "bits": 100,
            },
            {
                "query": "g1|p0",
                "target": "g0|p0",
                "pident": 95.0,
                "alnlen": 3,
                "qlen": 3,
                "tlen": 3,
                "evalue": 1e-20,
                "bits": 100,
            },
            {
                "query": "g0|p1",
                "target": "g1|p1",
                "pident": 80.0,
                "alnlen": 3,
                "qlen": 3,
                "tlen": 3,
                "evalue": 1e-10,
                "bits": 50,
            },
        ]
    )
    filtered_hits = filter_mmseqs_hits(add_protein_id_columns(hits))
    pairwise = compute_pairwise_aai(prepared, filtered_hits, min_alignment_fraction=0.2)
    aai_matrix, distance_matrix = build_aai_distance_matrices(pairwise, prepared["genome_id"].tolist())

    ab = pairwise[(pairwise["genome_a"] == "genome_a") & (pairwise["genome_b"] == "genome_b")].iloc[0]
    ac = pairwise[(pairwise["genome_a"] == "genome_a") & (pairwise["genome_b"] == "genome_c")].iloc[0]

    assert ab["rbh_count"] == 1
    assert ab["aai"] == 95.0
    assert ab["af"] == 0.5
    assert bool(ab["valid"]) is True
    assert ac["rbh_count"] == 0
    assert bool(ac["valid"]) is False
    assert aai_matrix.loc["genome_a", "genome_b"] == 95.0
    assert distance_matrix[0, 1] == pytest.approx(0.05)
    assert distance_matrix[0, 2] == 1.0


def test_clustering_metrics_recover_synthetic_species_clusters():
    distance_matrix = np.array(
        [
            [0.0, 0.05, 0.95, 0.95],
            [0.05, 0.0, 0.95, 0.95],
            [0.95, 0.95, 0.0, 0.04],
            [0.95, 0.95, 0.04, 0.0],
        ]
    )
    species = ["species_a", "species_a", "species_b", "species_b"]
    clusters = cluster_distance_matrix(distance_matrix, n_clusters=2)
    metrics = evaluate_species_clustering(species, clusters, distance_matrix)

    assert metrics["ari"] == 1.0
    assert metrics["nmi"] == 1.0


def test_leiden_aai_metrics_runs_parameter_grid_on_precomputed_distances():
    distance_matrix = np.array(
        [
            [0.0, 0.05, 0.95, 0.95],
            [0.05, 0.0, 0.95, 0.95],
            [0.95, 0.95, 0.0, 0.04],
            [0.95, 0.95, 0.04, 0.0],
        ]
    )
    genome_index = pd.DataFrame(
        {
            "genome_id": ["a1", "a2", "b1", "b2"],
            "species": ["species_a", "species_a", "species_b", "species_b"],
        }
    )

    clusters, metrics = compute_leiden_aai_metrics(
        distance_matrix=distance_matrix,
        genome_index=genome_index,
        leiden_resolutions=[0.1, 0.25, 1.0],
        k_neighbors=[2],
        label_col="species",
        show_progress=False,
    )

    assert len(metrics) == 3
    assert set(metrics["resolution"]) == {0.1, 0.25, 1.0}
    assert set(metrics["k_neighbors"]) == {2}
    assert set(clusters.columns) == {
        "genome_id",
        "species",
        "leiden_clusters",
        "resolution",
        "k_neighbors",
        "effective_k_neighbors",
    }


def test_rank_leiden_metrics_and_select_final_clusters():
    metrics = pd.DataFrame(
        {
            "ari": [0.2, 0.8, 0.8],
            "nmi": [0.4, 0.7, 0.9],
            "v_measure": [0.4, 0.7, 0.9],
            "silhouette": [0.1, 0.2, 0.3],
            "resolution": [0.1, 0.25, 1.0],
            "k_neighbors": [5, 10, 15],
        }
    )
    clusters = pd.DataFrame(
        {
            "genome_id": ["a", "b", "a", "b"],
            "species": ["s1", "s2", "s1", "s2"],
            "leiden_clusters": ["0", "1", "0", "0"],
            "resolution": [0.25, 0.25, 1.0, 1.0],
            "k_neighbors": [10, 10, 15, 15],
            "effective_k_neighbors": [10, 10, 15, 15],
        }
    )

    ranked = rank_leiden_metrics(metrics)
    final_clusters = select_final_clusters(clusters, ranked.iloc[0])

    assert ranked["rank"].tolist() == [1, 2, 3]
    assert ranked.iloc[0]["resolution"] == 1.0
    assert ranked.iloc[0]["k_neighbors"] == 15
    assert final_clusters["genome_id"].tolist() == ["a", "b"]
    assert final_clusters["resolution"].tolist() == [1.0, 1.0]
