import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from tap import Tap
from tqdm import tqdm

FILENAMES = [
    ("Bacformer", "bacformer_genome_embeddings.parquet"),
    ("ESM-2", "esm2_genome_embeddings.parquet"),
    ("ESMC", "esmc_genome_embeddings.parquet"),
    ("Nucleotide Transformer", "nucleotide_transformer_genome_embeddings.parquet"),
    ("Mistral-DNA", "mistral_genome_embeddings.parquet"),
    ("ProtBert", "protbert_genome_embeddings.parquet"),
    ("DNABERT-2", "dnabert_genome_embeddings.parquet"),
]


def leiden_clustering(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_neighbors: int = 10,
    resolution: float = 1.0,
):
    """
    Create an AnnData object from embeddings (X) and metadata, then run Leiden clustering.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n_samples, n_features)
        Embedding matrix for your samples.
    metadata : dict or pd.DataFrame
        Per-genome metadata.
        Length of metadata must match n_samples in `embeddings`.
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering.
    n_neighbors : int, default=10
        Number of neighbors to construct the neighborhood graph in Scanpy.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object containing embeddings and clustering results.
    """
    # 1. Create AnnData object from embeddings
    adata = anndata.AnnData(X=embeddings.copy())

    # Make sure 'metadata' is a pandas DataFrame if it's a dict
    if isinstance(metadata, dict):
        metadata = pd.DataFrame(metadata)
    elif not isinstance(metadata, pd.DataFrame):
        raise ValueError("metadata must be either a dict or pandas DataFrame.")

    # Assign metadata to adata.obs
    # (Ensure the length of metadata matches the number of rows in `embeddings`)
    adata.obs = metadata.reset_index(drop=True)

    # 2. Compute nearest neighbors and run Leiden clustering
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")  # use the 'X' matrix as input
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden_clusters")
    return adata


def compute_leiden_clustering_metrics(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_neighbors: int = 10,
    resolution: float = 1.0,
    label_key: str = "species",
):
    """
    Create an AnnData object from embeddings (X) and metadata, then run Leiden clustering.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n_samples, n_features)
        Embedding matrix for your samples.
    metadata : dict or pd.DataFrame
        Per-genome metadata.
        Length of metadata must match n_samples in `embeddings`.
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering.
    n_neighbors : int, default=10
        Number of neighbors to construct the neighborhood graph in Scanpy.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object containing embeddings and clustering results.
    ari : float
        Adjusted Rand Index.
    nmi : float
        Normalized Mutual Information.
    sil : float
        Silhouette Score.
    """
    adata = leiden_clustering(
        embeddings=embeddings,
        metadata=metadata,
        n_neighbors=n_neighbors,
        resolution=resolution,
    )

    # Convert Leiden cluster labels to integer labels
    leiden_clusters = adata.obs["leiden_clusters"].astype(int)

    # 3. Encode your ground-truth labels
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(adata.obs[label_key])

    # 4. Compute ARI, NMI, and Silhouette
    ari = adjusted_rand_score(numeric_labels, leiden_clusters)
    nmi = normalized_mutual_info_score(numeric_labels, leiden_clusters)
    # Silhouette requires sample-level features + predicted labels
    sil = silhouette_score(adata.X, leiden_clusters)

    print(f"Leiden clustering at resolution={resolution}")
    print(f"  Adjusted Rand Index (ARI): {ari}")
    print(f"  Normalized Mutual Information (NMI): {nmi}")
    print(f"  Silhouette Score: {sil}")

    return adata, ari, nmi, sil


def run(
    input_dir: str,
    output_dir: str,
    filenames: list[tuple[str, str]] = FILENAMES,
    leiden_resolutions: list[float] = None,
    k_neighbors: list[int] = None,
):
    """Run the script."""
    if k_neighbors is None:
        k_neighbors = [5, 10, 15]
    if leiden_resolutions is None:
        leiden_resolutions = [0.1, 0.25, 0.5, 1.0]
    output = []
    for model_name, f in tqdm(filenames):
        df = pd.read_parquet(os.path.join(input_dir, f))
        embeds = np.stack(df["genome_embedding"].tolist())
        for resolution in tqdm(leiden_resolutions):
            for k in k_neighbors:
                print(f"Running {model_name} with leiden resolution {resolution} and k neighbors {k}")

                _, ari, nmi, sil = compute_leiden_clustering_metrics(
                    embeddings=embeds,
                    metadata=df,
                    n_neighbors=k,
                    resolution=resolution,
                    label_key="species",
                )
                output.append(
                    {
                        "ARI": ari,
                        "NMI": nmi,
                        "ASW": sil,
                        "Model": model_name,
                        "resolution": resolution,
                        "k_neighbors": k,
                        "taxa_level": "species",
                    }
                )

                _, ari, nmi, sil = compute_leiden_clustering_metrics(
                    embeddings=embeds,
                    metadata=df,
                    n_neighbors=k,
                    resolution=resolution,
                    label_key="genus",
                )
                output.append(
                    {
                        "ARI": ari,
                        "NMI": nmi,
                        "ASW": sil,
                        "Model": model_name,
                        "resolution": resolution,
                        "k_neighbors": k,
                        "taxa_level": "genus",
                    }
                )

                _, ari, nmi, sil = compute_leiden_clustering_metrics(
                    embeddings=embeds,
                    metadata=df,
                    n_neighbors=k,
                    resolution=resolution,
                    label_key="family",
                )
                output.append(
                    {
                        "ARI": ari,
                        "NMI": nmi,
                        "ASW": sil,
                        "Model": model_name,
                        "resolution": resolution,
                        "k_neighbors": k,
                        "taxa_level": "family",
                    }
                )

            output_df = pd.DataFrame(output)
            output_df.to_parquet(os.path.join(output_dir, "strain_clustering_eval_results.parquet"))


class ArgumentParser(Tap):
    """Argument parser for training Bacformer (Lightning version)."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str
    output_dir: str
    filenames: list[tuple[str, str]] = FILENAMES
    leiden_resolutions: list[float] = [0.1, 0.25, 0.5, 1.0]
    k_neighbors: list[int] = [5, 10, 15]


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filenames=args.filenames,
        leiden_resolutions=args.leiden_resolutions,
        k_neighbors=args.k_neighbors,
    )
