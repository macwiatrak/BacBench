import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from tap import Tap
from tqdm import tqdm


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
    input_df_file_path: str,
    output_dir: str | None = None,
    model_name: str = "Unknown_model",
    leiden_resolutions: list[float] | None = None,
    k_neighbors: list[int] | None = None,
    input_col: str = "embeddings",
    n_bootstraps: int = 10,
    proportion: float = 0.8,
    random_seed: int = 42,
):
    """Run the script with bootstrapping for robust metric estimation.

    Parameters
    ----------
    input_df_file_path : str
        Path to input parquet file with embeddings.
    output_dir : str | None
        Directory to save results.
    model_name : str
        Name of the model being evaluated.
    leiden_resolutions : list[float] | None
        Resolutions for Leiden clustering.
    k_neighbors : list[int] | None
        Numbers of neighbors for graph construction.
    input_col : str
        Column name containing embeddings.
    n_bootstraps : int
        Number of bootstrap samples.
    proportion : float
        Proportion of data to sample in each bootstrap (0 < proportion <= 1).
    random_seed : int
        Random seed for reproducibility.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    if k_neighbors is None:
        k_neighbors = [5, 10, 15]
    if leiden_resolutions is None:
        leiden_resolutions = [0.1, 0.25, 0.5, 1.0]
    output = []

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    df = pd.read_parquet(input_df_file_path)
    input_col = "embeddings" if "embeddings" in df.columns else "genome_embedding"
    embeds = np.stack(df[input_col].tolist())
    n_samples = len(embeds)
    sample_size = int(n_samples * proportion)

    for res in tqdm(leiden_resolutions, desc="Leiden resolutions"):
        for k in k_neighbors:
            print(f"Running {model_name} with leiden resolution {res} and k neighbors {k}")

            for bootstrap_idx in tqdm(range(n_bootstraps), desc=f"Bootstrap samples (res={res}, k={k})", leave=False):
                # Sample with replacement
                indices = np.random.choice(n_samples, size=sample_size, replace=True)
                bootstrap_embeds = embeds[indices]
                bootstrap_df = df.iloc[indices].reset_index(drop=True)

                # Compute metrics for this bootstrap sample
                _, ari, nmi, sil = compute_leiden_clustering_metrics(
                    embeddings=bootstrap_embeds,
                    metadata=bootstrap_df,
                    n_neighbors=k,
                    resolution=res,
                    label_key="species",
                )

                # Compute mean and std from bootstrap samples
                output.append(
                    {
                        "ARI": ari,
                        "NMI": nmi,
                        "ASW_mean": sil,
                        "Model": model_name,
                        "resolution": res,
                        "k_neighbors": k,
                        "taxa_level": "species",
                        "bootstrap_idx": bootstrap_idx,
                        "random_seed": random_seed,
                    }
                )

                _, ari, nmi, sil = compute_leiden_clustering_metrics(
                    embeddings=embeds,
                    metadata=df,
                    n_neighbors=k,
                    resolution=res,
                    label_key="genus",
                )
                output.append(
                    {
                        "ARI": ari,
                        "NMI": nmi,
                        "ASW": sil,
                        "Model": model_name,
                        "resolution": res,
                        "k_neighbors": k,
                        "taxa_level": "genus",
                    }
                )

                _, ari, nmi, sil = compute_leiden_clustering_metrics(
                    embeddings=embeds,
                    metadata=df,
                    n_neighbors=k,
                    resolution=res,
                    label_key="family",
                )
                output.append(
                    {
                        "ARI": ari,
                        "NMI": nmi,
                        "ASW": sil,
                        "Model": model_name,
                        "resolution": res,
                        "k_neighbors": k,
                        "taxa_level": "family",
                    }
                )

    output_df = pd.DataFrame(output)
    if output_dir is not None:
        output_df.to_parquet(os.path.join(output_dir, f"results_{model_name}.parquet"))


class ArgumentParser(Tap):
    """Argument parser for strain clustering evaluation with bootstrapping."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_df_filepath: str
    output_dir: str | None
    model_name: str
    leiden_resolutions: list[float] = [0.1, 0.25, 0.5]
    k_neighbors: list[int] = [5, 10, 15]
    input_col: str = "embeddings"
    n_bootstraps: int = 10  # Number of bootstrap samples
    proportion: float = 0.8  # Proportion of data to sample in each bootstrap
    random_seed: int = 42  # Random seed for reproducibility


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    res_df = run(
        input_df_file_path=args.input_df_filepath,
        output_dir=args.output_dir,
        model_name=args.model_name,
        leiden_resolutions=args.leiden_resolutions,
        k_neighbors=args.k_neighbors,
        input_col=args.input_col,
        n_bootstraps=args.n_bootstraps,
        proportion=args.proportion,
        random_seed=args.random_seed,
    )
