import csv
import inspect
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from tqdm import tqdm

MMSEQS_COLUMNS = ["query", "target", "pident", "alnlen", "qlen", "tlen", "evalue", "bits"]


def flatten_protein_sequences(protein_sequences: object) -> list[str]:
    """Flatten a genome protein sequence field into a clean list of proteins."""
    proteins: list[str] = []

    def collect(value: object) -> None:
        if value is None:
            return
        if isinstance(value, float) and np.isnan(value):
            return
        if isinstance(value, str):
            sequence = value.strip()
            if sequence:
                proteins.append(sequence)
            return
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                collect(value.item())
            else:
                for item in value.tolist():
                    collect(item)
            return
        if isinstance(value, list | tuple):
            for item in value:
                collect(item)
            return
        raise TypeError(f"Unsupported protein sequence value of type {type(value).__name__}")

    collect(protein_sequences)
    return proteins


def prepare_genome_dataframe(
    df: pd.DataFrame,
    genome_col: str = "genome_id",
    proteins_col: str = "protein_sequence",
    species_col: str = "species",
    min_species_count: int = 2,
) -> pd.DataFrame:
    """Validate and normalize the genome dataframe used for AAI clustering."""
    if proteins_col not in df.columns:
        raise ValueError(f"Input dataframe must contain a {proteins_col!r} column.")
    if species_col not in df.columns:
        raise ValueError(f"Input dataframe must contain a {species_col!r} column.")

    prepared = df.copy()
    if genome_col not in prepared.columns:
        prepared[genome_col] = prepared.index.astype(str)

    prepared = prepared[[genome_col, proteins_col, species_col]].dropna(subset=[genome_col, species_col])
    prepared = prepared.drop_duplicates(subset=[genome_col], keep="first").reset_index(drop=True)
    prepared[proteins_col] = prepared[proteins_col].apply(flatten_protein_sequences)
    prepared["n_proteins"] = prepared[proteins_col].str.len()
    prepared = prepared[prepared["n_proteins"] > 0].reset_index(drop=True)

    if min_species_count > 1 and not prepared.empty:
        species_counts = prepared[species_col].value_counts()
        retained_species = species_counts[species_counts >= min_species_count].index
        prepared = prepared[prepared[species_col].isin(retained_species)].reset_index(drop=True)

    if prepared.empty:
        raise ValueError("No genomes remain after filtering missing labels, duplicate IDs, and empty protein lists.")

    prepared = prepared.rename(
        columns={
            genome_col: "genome_id",
            proteins_col: "protein_sequence",
            species_col: "species",
        }
    )
    prepared["genome_idx"] = np.arange(len(prepared), dtype=int)
    return prepared[["genome_idx", "genome_id", "species", "protein_sequence", "n_proteins"]]


def make_protein_id(genome_idx: int, protein_idx: int) -> str:
    """Create the stable protein identifier used in FASTA and MMseqs outputs."""
    return f"g{genome_idx}|p{protein_idx}"


def parse_protein_id(protein_id: str) -> tuple[int, int]:
    """Parse a protein identifier created by `make_protein_id`."""
    genome_part, protein_part = protein_id.split("|", maxsplit=1)
    if not genome_part.startswith("g") or not protein_part.startswith("p"):
        raise ValueError(f"Protein ID {protein_id!r} does not match the expected g<idx>|p<idx> format.")
    return int(genome_part[1:]), int(protein_part[1:])


def write_protein_fasta(prepared_df: pd.DataFrame, fasta_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Write all genome proteins to FASTA and return the protein ID mapping table."""
    fasta_path = Path(fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    protein_rows = []

    with fasta_path.open("w") as handle:
        for row in prepared_df.itertuples(index=False):
            for protein_idx, sequence in enumerate(row.protein_sequence):
                protein_id = make_protein_id(int(row.genome_idx), protein_idx)
                handle.write(f">{protein_id}\n")
                handle.write(f"{sequence}\n")
                protein_rows.append(
                    {
                        "protein_id": protein_id,
                        "genome_idx": int(row.genome_idx),
                        "genome_id": row.genome_id,
                        "protein_idx": protein_idx,
                    }
                )

    return pd.DataFrame(protein_rows)


def _iter_input_batches(
    input_df_filepath: str | os.PathLike[str],
    columns: list[str],
    batch_size: int,
) -> Iterator[tuple[pd.DataFrame, list[str]]]:
    """Yield input rows in bounded-size batches."""
    input_df_filepath = Path(input_df_filepath)
    if input_df_filepath.suffix == ".csv":
        for batch in pd.read_csv(input_df_filepath, usecols=lambda column: column in columns, chunksize=batch_size):
            yield batch, batch.columns.tolist()
        return

    parquet_file = pq.ParquetFile(input_df_filepath)
    available_columns = parquet_file.schema_arrow.names
    read_columns = [column for column in columns if column in available_columns]
    for batch in parquet_file.iter_batches(columns=read_columns, batch_size=batch_size):
        yield batch.to_pandas(), read_columns


def _get_input_num_rows(input_df_filepath: str | os.PathLike[str]) -> int | None:
    """Return the number of input rows when it can be read cheaply."""
    input_df_filepath = Path(input_df_filepath)
    if input_df_filepath.suffix == ".csv":
        return None
    return pq.ParquetFile(input_df_filepath).metadata.num_rows


def _validate_streamed_columns(
    input_df_filepath: str | os.PathLike[str],
    genome_col: str,
    proteins_col: str,
    species_col: str,
) -> bool:
    """Validate the streamed input schema and return whether `genome_col` exists."""
    input_df_filepath = Path(input_df_filepath)
    if input_df_filepath.suffix == ".csv":
        available_columns = pd.read_csv(input_df_filepath, nrows=0).columns.tolist()
    else:
        available_columns = pq.ParquetFile(input_df_filepath).schema_arrow.names

    missing_columns = [column for column in [proteins_col, species_col] if column not in available_columns]
    if missing_columns:
        raise ValueError(f"Input dataframe is missing required columns: {missing_columns}.")
    return genome_col in available_columns


def _iter_valid_genome_records(
    input_df_filepath: str | os.PathLike[str],
    genome_col: str,
    proteins_col: str,
    species_col: str,
    input_batch_size: int,
    progress_desc: str | None = None,
    show_progress: bool = True,
) -> Iterator[tuple[str, str, list[str]]]:
    """Yield valid genome records with flattened protein sequences."""
    has_genome_col = _validate_streamed_columns(input_df_filepath, genome_col, proteins_col, species_col)
    columns = [proteins_col, species_col]
    if has_genome_col:
        columns.insert(0, genome_col)

    row_offset = 0
    progress = tqdm(
        total=_get_input_num_rows(input_df_filepath),
        desc=progress_desc,
        unit="rows",
        disable=not show_progress or progress_desc is None,
    )
    try:
        for batch, _ in _iter_input_batches(input_df_filepath, columns=columns, batch_size=input_batch_size):
            batch = batch.reset_index(drop=True)
            for row_idx, row in batch.iterrows():
                genome_id = row[genome_col] if has_genome_col else str(row_offset + row_idx)
                species = row[species_col]
                if pd.isna(genome_id) or pd.isna(species):
                    continue

                proteins = flatten_protein_sequences(row[proteins_col])
                if proteins:
                    yield str(genome_id), str(species), proteins
            row_offset += len(batch)
            progress.update(len(batch))
    finally:
        progress.close()


def prepare_genomes_from_input(
    input_df_filepath: str | os.PathLike[str],
    fasta_path: str | os.PathLike[str],
    protein_index_path: str | os.PathLike[str],
    genome_col: str = "genome_name",
    proteins_col: str = "protein_sequence",
    species_col: str = "species",
    min_species_count: int = 2,
    input_batch_size: int = 100,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Stream the genome table, write FASTA/protein index files, and return genome metadata.

    Large strain-clustering protein-sequence parquets cannot be loaded with
    `pd.read_parquet` on some systems. This helper reads with
    `pyarrow.parquet.ParquetFile.iter_batches`, flattens each genome's
    contig-grouped proteins one row at a time, and writes the protein FASTA and
    protein index incrementally.
    """
    species_counts: Counter[str] = Counter()
    valid_genome_ids = set()
    for genome_id, species, _ in _iter_valid_genome_records(
        input_df_filepath=input_df_filepath,
        genome_col=genome_col,
        proteins_col=proteins_col,
        species_col=species_col,
        input_batch_size=input_batch_size,
        progress_desc="Counting valid genomes",
        show_progress=show_progress,
    ):
        if genome_id in valid_genome_ids:
            continue
        valid_genome_ids.add(genome_id)
        species_counts[species] += 1

    if min_species_count > 1:
        retained_species = {species for species, count in species_counts.items() if count >= min_species_count}
    else:
        retained_species = set(species_counts)

    fasta_path = Path(fasta_path)
    protein_index_path = Path(protein_index_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    protein_index_path.parent.mkdir(parents=True, exist_ok=True)

    genome_rows = []
    written_genome_ids = set()
    genome_idx = 0
    with fasta_path.open("w") as fasta_handle, protein_index_path.open("w", newline="") as protein_index_handle:
        protein_index_writer = csv.DictWriter(
            protein_index_handle,
            fieldnames=["protein_id", "genome_idx", "genome_id", "protein_idx"],
        )
        protein_index_writer.writeheader()
        for genome_id, species, proteins in _iter_valid_genome_records(
            input_df_filepath=input_df_filepath,
            genome_col=genome_col,
            proteins_col=proteins_col,
            species_col=species_col,
            input_batch_size=input_batch_size,
            progress_desc="Writing protein FASTA",
            show_progress=show_progress,
        ):
            if genome_id in written_genome_ids or species not in retained_species:
                continue

            for protein_idx, sequence in enumerate(proteins):
                protein_id = make_protein_id(genome_idx, protein_idx)
                fasta_handle.write(f">{protein_id}\n{sequence}\n")
                protein_index_writer.writerow(
                    {
                        "protein_id": protein_id,
                        "genome_idx": genome_idx,
                        "genome_id": genome_id,
                        "protein_idx": protein_idx,
                    }
                )

            genome_rows.append(
                {
                    "genome_idx": genome_idx,
                    "genome_id": genome_id,
                    "species": species,
                    "n_proteins": len(proteins),
                }
            )
            written_genome_ids.add(genome_id)
            genome_idx += 1

    prepared_df = pd.DataFrame(genome_rows)
    if prepared_df.empty:
        raise ValueError("No genomes remain after filtering missing labels, duplicate IDs, and empty protein lists.")

    return prepared_df


def run_mmseqs_all_vs_all(
    fasta_path: str | os.PathLike[str],
    output_tsv_path: str | os.PathLike[str],
    tmp_dir: str | os.PathLike[str],
    mmseqs_binary: str = "mmseqs",
    threads: int | None = 32,
    split_memory_limit: str | None = "110G",
    max_evalue: float | None = 1e-5,
    min_coverage: float | None = 0.5,
    coverage_mode: int = 0,
    max_seqs: int | None = 50,
    force: bool = False,
) -> Path:
    """Run an MMseqs2 all-vs-all protein search and return the tabular output path."""
    fasta_path = Path(fasta_path)
    output_tsv_path = Path(output_tsv_path)
    tmp_dir = Path(tmp_dir)

    if output_tsv_path.exists() and not force:
        return output_tsv_path

    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        mmseqs_binary,
        "easy-search",
        str(fasta_path),
        str(fasta_path),
        str(output_tsv_path),
        str(tmp_dir),
        "--format-output",
        ",".join(MMSEQS_COLUMNS),
        "--remove-tmp-files",
        "1",
    ]
    if threads is not None:
        command.extend(["--threads", str(threads)])
    if split_memory_limit is not None:
        command.extend(["--split-memory-limit", split_memory_limit])
    if max_evalue is not None:
        command.extend(["-e", str(max_evalue)])
    if min_coverage is not None:
        command.extend(["-c", str(min_coverage), "--cov-mode", str(coverage_mode)])
    if max_seqs is not None:
        command.extend(["--max-seqs", str(max_seqs)])

    subprocess.run(command, check=True)
    return output_tsv_path


def read_mmseqs_hits(hits_tsv_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read MMseqs2 tabular hits produced by `run_mmseqs_all_vs_all`."""
    hits_tsv_path = Path(hits_tsv_path)
    if hits_tsv_path.stat().st_size == 0:
        return pd.DataFrame(columns=MMSEQS_COLUMNS)
    return pd.read_csv(hits_tsv_path, sep="\t", names=MMSEQS_COLUMNS)


def add_protein_id_columns(hits: pd.DataFrame) -> pd.DataFrame:
    """Add genome/protein index columns parsed from MMseqs query and target IDs."""
    if hits.empty:
        output = hits.copy()
        for column in ["query_genome_idx", "query_protein_idx", "target_genome_idx", "target_protein_idx"]:
            output[column] = pd.Series(dtype="int64")
        return output

    query_parts = hits["query"].str.extract(r"^g(?P<genome>\d+)\|p(?P<protein>\d+)$")
    target_parts = hits["target"].str.extract(r"^g(?P<genome>\d+)\|p(?P<protein>\d+)$")
    if query_parts.isna().any().any() or target_parts.isna().any().any():
        raise ValueError("MMseqs hits contain protein IDs that do not match the expected g<idx>|p<idx> format.")

    output = hits.copy()
    output["query_genome_idx"] = query_parts["genome"].astype(int)
    output["query_protein_idx"] = query_parts["protein"].astype(int)
    output["target_genome_idx"] = target_parts["genome"].astype(int)
    output["target_protein_idx"] = target_parts["protein"].astype(int)
    return output


def filter_mmseqs_hits(
    hits: pd.DataFrame,
    max_evalue: float = 1e-5,
    min_query_coverage: float = 0.5,
    min_target_coverage: float = 0.5,
    exclude_same_genome: bool = True,
) -> pd.DataFrame:
    """Filter MMseqs hits by e-value, alignment coverage, and same-genome matches."""
    if hits.empty:
        return hits.copy()

    output = hits.copy()
    for column in ["pident", "alnlen", "qlen", "tlen", "evalue", "bits"]:
        output[column] = pd.to_numeric(output[column])
    output["query_coverage"] = output["alnlen"] / output["qlen"]
    output["target_coverage"] = output["alnlen"] / output["tlen"]

    mask = (
        (output["evalue"] <= max_evalue)
        & (output["query_coverage"] >= min_query_coverage)
        & (output["target_coverage"] >= min_target_coverage)
    )
    if exclude_same_genome:
        mask &= output["query_genome_idx"] != output["target_genome_idx"]
    return output[mask].reset_index(drop=True)


def compute_pairwise_aai(
    prepared_df: pd.DataFrame,
    filtered_hits: pd.DataFrame,
    min_alignment_fraction: float = 0.2,
) -> pd.DataFrame:
    """Compute pairwise AAI from filtered MMseqs hits using reciprocal best hits."""
    protein_counts = prepared_df.set_index("genome_idx")["n_proteins"].to_dict()
    genome_ids = prepared_df.set_index("genome_idx")["genome_id"].to_dict()
    pair_rows = {
        (int(i), int(j)): {
            "genome_a_idx": int(i),
            "genome_b_idx": int(j),
            "genome_a": genome_ids[int(i)],
            "genome_b": genome_ids[int(j)],
            "n_proteins_a": int(protein_counts[int(i)]),
            "n_proteins_b": int(protein_counts[int(j)]),
            "aai": np.nan,
            "af": 0.0,
            "rbh_count": 0,
            "valid": False,
        }
        for i in prepared_df["genome_idx"]
        for j in prepared_df["genome_idx"]
        if int(i) < int(j)
    }

    if filtered_hits.empty:
        return pd.DataFrame(pair_rows.values())

    sort_columns = ["query", "target_genome_idx", "bits", "pident", "evalue"]
    sorted_hits = filtered_hits.sort_values(
        sort_columns,
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    best_hits = sorted_hits.groupby(["query", "target_genome_idx"], as_index=False, sort=False).head(1)
    reciprocal_hits = best_hits.merge(
        best_hits,
        left_on=["query", "target"],
        right_on=["target", "query"],
        suffixes=("_forward", "_reverse"),
    )
    reciprocal_hits = reciprocal_hits[
        reciprocal_hits["query_genome_idx_forward"] < reciprocal_hits["target_genome_idx_forward"]
    ].copy()

    if not reciprocal_hits.empty:
        reciprocal_hits["rbh_pident"] = (reciprocal_hits["pident_forward"] + reciprocal_hits["pident_reverse"]) / 2
        grouped = reciprocal_hits.groupby(
            ["query_genome_idx_forward", "target_genome_idx_forward"], as_index=False
        ).agg(
            aai=("rbh_pident", "mean"),
            rbh_count=("rbh_pident", "size"),
        )

        for row in grouped.itertuples(index=False):
            genome_a_idx = int(row.query_genome_idx_forward)
            genome_b_idx = int(row.target_genome_idx_forward)
            key = (genome_a_idx, genome_b_idx)
            alignment_fraction = row.rbh_count / min(protein_counts[genome_a_idx], protein_counts[genome_b_idx])
            pair_rows[key]["aai"] = float(row.aai)
            pair_rows[key]["af"] = float(alignment_fraction)
            pair_rows[key]["rbh_count"] = int(row.rbh_count)
            pair_rows[key]["valid"] = bool(row.rbh_count > 0 and alignment_fraction >= min_alignment_fraction)

    return pd.DataFrame(pair_rows.values())


def build_aai_distance_matrices(pairwise_aai: pd.DataFrame, genome_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Build an AAI matrix and clustering distance matrix from pairwise AAI rows."""
    n_genomes = len(genome_ids)
    aai_values = np.full((n_genomes, n_genomes), np.nan, dtype=float)
    distance_values = np.ones((n_genomes, n_genomes), dtype=float)
    np.fill_diagonal(aai_values, 100.0)
    np.fill_diagonal(distance_values, 0.0)

    for row in pairwise_aai.itertuples(index=False):
        i = int(row.genome_a_idx)
        j = int(row.genome_b_idx)
        if bool(row.valid) and not pd.isna(row.aai):
            aai_values[i, j] = float(row.aai)
            aai_values[j, i] = float(row.aai)
            distance = 1 - (float(row.aai) / 100)
            distance_values[i, j] = distance
            distance_values[j, i] = distance

    aai_matrix = pd.DataFrame(aai_values, index=genome_ids, columns=genome_ids)
    return aai_matrix, distance_values


def cluster_distance_matrix(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster genomes from a precomputed distance matrix."""
    if n_clusters < 2:
        raise ValueError("AAI clustering requires at least two clusters.")
    if distance_matrix.shape[0] < n_clusters:
        raise ValueError("Number of clusters cannot exceed the number of genomes.")

    clustering_kwargs = {
        "n_clusters": n_clusters,
        "linkage": "average",
    }
    if "metric" in inspect.signature(AgglomerativeClustering).parameters:
        clustering_kwargs["metric"] = "precomputed"
    else:
        clustering_kwargs["affinity"] = "precomputed"

    model = AgglomerativeClustering(**clustering_kwargs)
    return model.fit_predict(distance_matrix)


def evaluate_species_clustering(
    species: pd.Series | list[str],
    cluster_labels: np.ndarray,
    distance_matrix: np.ndarray,
) -> dict[str, float]:
    """Compute species-clustering metrics from ground truth and predicted clusters."""
    species_values = pd.Series(species).astype(str).to_numpy()
    unique_clusters = np.unique(cluster_labels)
    if len(unique_clusters) > 1 and len(unique_clusters) < len(cluster_labels):
        silhouette = float(silhouette_score(distance_matrix, cluster_labels, metric="precomputed"))
    else:
        silhouette = np.nan

    return {
        "ari": float(adjusted_rand_score(species_values, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(species_values, cluster_labels)),
        "homogeneity": float(homogeneity_score(species_values, cluster_labels)),
        "completeness": float(completeness_score(species_values, cluster_labels)),
        "v_measure": float(v_measure_score(species_values, cluster_labels)),
        "silhouette": silhouette,
        "n_genomes": int(len(species_values)),
        "n_species": int(pd.Series(species_values).nunique()),
        "n_clusters": int(len(unique_clusters)),
    }


def leiden_clustering_from_distance(
    distance_matrix: np.ndarray,
    metadata: pd.DataFrame,
    n_neighbors: int = 10,
    resolution: float = 1.0,
) -> pd.DataFrame:
    """Run Leiden clustering from a precomputed AAI distance matrix."""
    import anndata
    import scanpy as sc

    if len(metadata) != distance_matrix.shape[0]:
        raise ValueError("Metadata length must match the distance matrix dimensions.")
    if len(metadata) < 2:
        raise ValueError("Leiden clustering requires at least two genomes.")

    effective_n_neighbors = min(n_neighbors, len(metadata) - 1)
    adata = anndata.AnnData(X=distance_matrix.copy())
    adata.obs = metadata.reset_index(drop=True).copy()
    sc.pp.neighbors(adata, n_neighbors=effective_n_neighbors, metric="precomputed")
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden_clusters")
    clusters = adata.obs[["genome_id", "species", "leiden_clusters"]].copy()
    clusters["leiden_clusters"] = clusters["leiden_clusters"].astype(str)
    clusters["resolution"] = resolution
    clusters["k_neighbors"] = n_neighbors
    clusters["effective_k_neighbors"] = effective_n_neighbors
    return clusters


def compute_leiden_aai_metrics(
    distance_matrix: np.ndarray,
    genome_index: pd.DataFrame,
    leiden_resolutions: list[float],
    k_neighbors: list[int],
    label_col: str = "species",
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Leiden clusters from an AAI distance matrix over a parameter grid."""
    if label_col not in genome_index.columns:
        raise ValueError(f"Genome index must contain a {label_col!r} label column.")

    parameter_grid = [(resolution, n_neighbors) for resolution in leiden_resolutions for n_neighbors in k_neighbors]
    all_clusters = []
    all_metrics = []
    for resolution, n_neighbors in tqdm(
        parameter_grid,
        desc="Running Leiden grid",
        unit="run",
        disable=not show_progress,
    ):
        clusters = leiden_clustering_from_distance(
            distance_matrix=distance_matrix,
            metadata=genome_index,
            n_neighbors=n_neighbors,
            resolution=resolution,
        )
        metrics = evaluate_species_clustering(
            species=genome_index[label_col],
            cluster_labels=clusters["leiden_clusters"].to_numpy(),
            distance_matrix=distance_matrix,
        )
        metrics.update(
            {
                "resolution": resolution,
                "k_neighbors": n_neighbors,
                "effective_k_neighbors": int(clusters["effective_k_neighbors"].iloc[0]),
                "label_col": label_col,
            }
        )
        all_clusters.append(clusters)
        all_metrics.append(metrics)

    return pd.concat(all_clusters, ignore_index=True), pd.DataFrame(all_metrics)


def rank_leiden_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Rank Leiden parameter settings by whole-dataset species clustering quality."""
    sort_columns = ["ari", "nmi", "v_measure", "silhouette", "resolution", "k_neighbors"]
    missing_columns = [column for column in sort_columns if column not in metrics_df.columns]
    if missing_columns:
        raise ValueError(f"Metrics dataframe is missing required ranking columns: {missing_columns}.")

    ranked = metrics_df.sort_values(
        by=sort_columns,
        ascending=[False, False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked


def select_final_clusters(clusters: pd.DataFrame, final_metrics: pd.Series) -> pd.DataFrame:
    """Select cluster assignments for the best-ranked Leiden setting."""
    return clusters[
        (clusters["resolution"] == final_metrics["resolution"])
        & (clusters["k_neighbors"] == final_metrics["k_neighbors"])
    ].reset_index(drop=True)


def plot_aai_heatmap(
    aai_matrix: pd.DataFrame,
    genome_index: pd.DataFrame,
    output_path: str | os.PathLike[str],
) -> None:
    """Write an AAI heatmap ordered by species."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    ordered = genome_index.sort_values(["species", "genome_id"])["genome_id"].tolist()
    ordered_aai = aai_matrix.loc[ordered, ordered]

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(ordered_aai.fillna(0).to_numpy(), vmin=0, vmax=100, cmap="viridis")
    ax.set_title("AAI matrix")
    ax.set_xlabel("Genome")
    ax.set_ylabel("Genome")
    if len(ordered) <= 40:
        ax.set_xticks(np.arange(len(ordered)))
        ax.set_xticklabels(ordered, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(len(ordered)))
        ax.set_yticklabels(ordered, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=ax, label="AAI")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_aai_dendrogram(
    distance_matrix: np.ndarray,
    genome_ids: list[str],
    output_path: str | os.PathLike[str],
) -> None:
    """Write an average-linkage dendrogram from the AAI distance matrix."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    if len(genome_ids) < 2:
        return

    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    fig_width = max(8, min(24, len(genome_ids) * 0.25))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    dendrogram(
        linkage_matrix,
        labels=genome_ids if len(genome_ids) <= 80 else None,
        leaf_rotation=90,
        leaf_font_size=6,
        ax=ax,
    )
    ax.set_ylabel("AAI distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mds_projection(
    distance_matrix: np.ndarray,
    genome_index: pd.DataFrame,
    output_path: str | os.PathLike[str],
    random_seed: int = 42,
) -> None:
    """Write a 2D MDS projection from the AAI distance matrix."""
    import matplotlib.pyplot as plt

    if len(genome_index) < 3:
        return

    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=random_seed).fit_transform(distance_matrix)
    species_codes = pd.Categorical(genome_index["species"])

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=species_codes.codes, cmap="tab20", s=20)
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.set_title("AAI distance projection")
    if len(species_codes.categories) <= 20:
        handles, _ = scatter.legend_elements()
        ax.legend(handles, species_codes.categories, title="Species", fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def read_input_dataframe(input_df_filepath: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a CSV or parquet dataframe from disk."""
    input_df_filepath = Path(input_df_filepath)
    if input_df_filepath.suffix == ".csv":
        return pd.read_csv(input_df_filepath)
    return pd.read_parquet(input_df_filepath)


def run_aai_clustering(
    input_df_filepath: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    genome_col: str = "genome_name",
    proteins_col: str = "protein_sequence",
    species_col: str = "species",
    min_species_count: int = 2,
    max_evalue: float = 1e-5,
    min_query_coverage: float = 0.5,
    min_target_coverage: float = 0.5,
    min_alignment_fraction: float = 0.2,
    mmseqs_binary: str = "mmseqs",
    mmseqs_split_memory_limit: str | None = "110G",
    mmseqs_max_seqs: int | None = 50,
    mmseqs_min_coverage: float | None = None,
    mmseqs_coverage_mode: int = 0,
    threads: int | None = 32,
    force: bool = False,
    make_plots: bool = True,
    random_seed: int = 42,
    input_batch_size: int = 100,
    leiden_resolutions: list[float] | None = None,
    k_neighbors: list[int] | None = None,
    label_col: str = "species",
    show_progress: bool = True,
) -> dict[str, object]:
    """Run the full AAI species clustering workflow."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if leiden_resolutions is None:
        leiden_resolutions = [0.1, 0.25, 1.0]
    if k_neighbors is None:
        k_neighbors = [5, 10, 15]

    parameters = {
        "input_df_filepath": str(input_df_filepath),
        "genome_col": genome_col,
        "proteins_col": proteins_col,
        "species_col": species_col,
        "min_species_count": min_species_count,
        "max_evalue": max_evalue,
        "min_query_coverage": min_query_coverage,
        "min_target_coverage": min_target_coverage,
        "min_alignment_fraction": min_alignment_fraction,
        "mmseqs_binary": mmseqs_binary,
        "mmseqs_split_memory_limit": mmseqs_split_memory_limit,
        "mmseqs_max_seqs": mmseqs_max_seqs,
        "mmseqs_min_coverage": mmseqs_min_coverage,
        "mmseqs_coverage_mode": mmseqs_coverage_mode,
        "threads": threads,
        "input_batch_size": input_batch_size,
        "leiden_resolutions": leiden_resolutions,
        "k_neighbors": k_neighbors,
        "label_col": label_col,
        "show_progress": show_progress,
    }
    (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2))

    if show_progress:
        tqdm.write("Preparing genome table and writing protein FASTA...")
    fasta_path = output_dir / "proteins.faa"
    prepared_df = prepare_genomes_from_input(
        input_df_filepath=input_df_filepath,
        fasta_path=fasta_path,
        protein_index_path=output_dir / "protein_index.csv",
        genome_col=genome_col,
        proteins_col=proteins_col,
        species_col=species_col,
        min_species_count=min_species_count,
        input_batch_size=input_batch_size,
        show_progress=show_progress,
    )
    genome_index = prepared_df[["genome_idx", "genome_id", "species", "n_proteins"]].copy()
    genome_index.to_csv(output_dir / "genome_index.csv", index=False)
    if show_progress:
        tqdm.write(
            f"Retained {len(genome_index):,} genomes across {genome_index['species'].nunique():,} species; "
            f"wrote {int(genome_index['n_proteins'].sum()):,} proteins."
        )

    if show_progress:
        tqdm.write("Running MMseqs2 all-vs-all protein search...")
    hits_tsv_path = run_mmseqs_all_vs_all(
        fasta_path=fasta_path,
        output_tsv_path=output_dir / "mmseqs_hits.tsv",
        tmp_dir=output_dir / "mmseqs_tmp",
        mmseqs_binary=mmseqs_binary,
        threads=threads,
        split_memory_limit=mmseqs_split_memory_limit,
        max_evalue=max_evalue,
        min_coverage=(
            min(min_query_coverage, min_target_coverage) if mmseqs_min_coverage is None else mmseqs_min_coverage
        ),
        coverage_mode=mmseqs_coverage_mode,
        max_seqs=mmseqs_max_seqs,
        force=force,
    )
    if show_progress:
        tqdm.write("Reading and filtering MMseqs2 hits...")
    hits = add_protein_id_columns(read_mmseqs_hits(hits_tsv_path))
    filtered_hits = filter_mmseqs_hits(
        hits=hits,
        max_evalue=max_evalue,
        min_query_coverage=min_query_coverage,
        min_target_coverage=min_target_coverage,
    )
    filtered_hits.to_parquet(output_dir / "filtered_mmseqs_hits.parquet", index=False)
    if show_progress:
        tqdm.write(f"Loaded {len(hits):,} MMseqs2 hits; retained {len(filtered_hits):,} after filtering.")

    if show_progress:
        tqdm.write("Computing reciprocal-best-hit AAI...")
    pairwise_aai = compute_pairwise_aai(
        prepared_df=prepared_df,
        filtered_hits=filtered_hits,
        min_alignment_fraction=min_alignment_fraction,
    )
    pairwise_aai.to_parquet(output_dir / "pairwise_aai.parquet", index=False)
    if show_progress:
        tqdm.write(
            f"Computed {len(pairwise_aai):,} genome pairs; "
            f"{int(pairwise_aai['valid'].sum()):,} passed the alignment-fraction threshold."
        )

    if show_progress:
        tqdm.write("Building AAI and distance matrices...")
    genome_ids = genome_index["genome_id"].tolist()
    aai_matrix, distance_matrix = build_aai_distance_matrices(pairwise_aai, genome_ids)
    aai_matrix.to_csv(output_dir / "aai_matrix.csv")
    np.save(output_dir / "distance_matrix.npy", distance_matrix)

    if show_progress:
        tqdm.write("Running Leiden clustering and evaluation grid...")
    clusters, metrics_df = compute_leiden_aai_metrics(
        distance_matrix=distance_matrix,
        genome_index=genome_index,
        leiden_resolutions=leiden_resolutions,
        k_neighbors=k_neighbors,
        label_col=label_col,
        show_progress=show_progress,
    )
    clusters.to_csv(output_dir / "clusters.csv", index=False)
    metrics_df["max_evalue"] = max_evalue
    metrics_df["min_query_coverage"] = min_query_coverage
    metrics_df["min_target_coverage"] = min_target_coverage
    metrics_df["min_alignment_fraction"] = min_alignment_fraction
    ranked_metrics = rank_leiden_metrics(metrics_df)
    final_metrics = ranked_metrics.iloc[[0]].copy()
    final_clusters = select_final_clusters(clusters, final_metrics.iloc[0])
    ranked_metrics.to_csv(output_dir / "metrics.csv", index=False)
    final_metrics.to_csv(output_dir / "final_metrics.csv", index=False)
    final_clusters.to_csv(output_dir / "final_clusters.csv", index=False)
    if show_progress:
        tqdm.write(
            "Best Leiden setting: "
            f"resolution={final_metrics['resolution'].iloc[0]}, "
            f"k_neighbors={final_metrics['k_neighbors'].iloc[0]}, "
            f"ARI={final_metrics['ari'].iloc[0]:.4f}, "
            f"NMI={final_metrics['nmi'].iloc[0]:.4f}."
        )

    if make_plots:
        if show_progress:
            tqdm.write("Writing AAI plots...")
        plot_aai_heatmap(aai_matrix, genome_index, output_dir / "aai_heatmap.png")
        plot_aai_dendrogram(distance_matrix, genome_ids, output_dir / "aai_dendrogram.png")
        plot_mds_projection(distance_matrix, genome_index, output_dir / "aai_mds.png", random_seed=random_seed)

    return {
        "prepared_df": prepared_df,
        "pairwise_aai": pairwise_aai,
        "aai_matrix": aai_matrix,
        "distance_matrix": distance_matrix,
        "clusters": clusters,
        "metrics": ranked_metrics,
        "final_metrics": final_metrics,
        "final_clusters": final_clusters,
    }
