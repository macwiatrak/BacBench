import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tap import Tap
from tqdm import tqdm


def mean_pairwise_cosine(arrays: list[np.ndarray]) -> float:
    """
    Compute the mean pairwise cosine similarity for a list of 1-D NumPy arrays.

    Parameters
    ----------
    arrays : list of np.ndarray
        All arrays must have the same length *N*; list length is *M*.

    Returns
    -------
    float
        Mean cosine similarity taken over the M·(M-1)/2 unique pairs.
    """
    if len(arrays) < 2:
        raise ValueError("Need at least two vectors to form a pair.")

    # Stack into shape (M, N) for one bulk call.
    X = np.vstack(arrays)  # shape (M, N)

    # Full similarity matrix (M × M, symmetric, ones on diagonal):
    sim = cosine_similarity(X)  # fast, handles normalisation internally

    # Grab the strictly upper triangular part (unique pairs) and average:
    i_upper = np.triu_indices_from(sim, k=1)
    score = sim[i_upper].mean()
    # normalize score to be between 0 and 1
    score = (score + 1) / 2
    return score


def run(input_df_filepath: str, output_dir: str, model_name: str = "bacformer", n_negatives: int = 10):
    """Run zero-shot operon identification evaluation"""
    os.makedirs(output_dir, exist_ok=True)
    # define a set of seeds for reproducibility when computing evaluation metrics
    random_seeds = [1, 12, 123, 1234, 12345]

    # read the data
    df = pd.read_parquet(input_df_filepath)
    # account for DNA LMs and protein LMs column names
    if "gene_embedding" in df.columns:
        df = df.rename(columns={"gene_embedding": "protein_embeddings"})
    else:
        # explode it by contig
        df = df.explode(
            ["contig_name", "operon_protein_names", "operon_protein_indices", "operon_names", "protein_embeddings"]
        )
    # separate protein embeddings from the rest of the data
    df_protein_embeddings = df[["contig_name", "protein_embeddings"]].set_index("contig_name")
    df = df.drop(columns=["protein_embeddings"])
    df = df.explode(["operon_protein_names", "operon_protein_indices", "operon_names"])
    # remove operons with nan values
    df = df.dropna(subset=["operon_protein_indices"])
    # get operon size
    df["operon_size"] = df.operon_protein_indices.apply(len)

    # run the evaluation for every operon
    output = defaultdict(list)
    for _, row in tqdm(df.iterrows()):
        # fetch contig protein embeddings
        protein_embeddings = df_protein_embeddings.loc[row["contig_name"]]["protein_embeddings"]
        # if model_type == "evo":
        #     protein_embeddings = fix_protein_embeddings_for_evo(protein_embeddings)
        # fetch operon protein indices
        operon_protein_indices = row["operon_protein_indices"]
        operon_prot_embeddings = [protein_embeddings[i] for i in operon_protein_indices]
        # compute operon scores by computing pairwise cosine similarity
        operon_score = mean_pairwise_cosine(operon_prot_embeddings)

        n_prot_embeds = len(protein_embeddings)
        operon_size = len(operon_protein_indices)

        for seed in random_seeds:
            random.seed(seed)
            # get random negatives
            random_negatives = random.sample(range(n_prot_embeds - operon_size), n_negatives)
            # get random negative indices
            random_negative_indices = [list(range(i, i + operon_size)) for i in random_negatives]
            # compute random negative scores
            random_negative_prot_embeds = [
                [protein_embeddings[i] for i in indices] for indices in random_negative_indices
            ]
            random_negative_scores = [
                mean_pairwise_cosine(neg_prot_embeds) for neg_prot_embeds in random_negative_prot_embeds
            ]
            # compute metrics
            y_true = np.array([1] + [0] * n_negatives)  # 1 positve, k negatives
            y_scores = np.array([operon_score] + random_negative_scores)

            auroc_val = roc_auc_score(y_true, y_scores)
            auprc_val = average_precision_score(y_true, y_scores)
            # store the results
            output["taxid"].append(row["taxid"])
            output["contig_name"].append(row["contig_name"])
            output["operon_name"].append(row["operon_names"])
            output["operon_size"].append(operon_size)
            output["operon_protein_indices"].append(operon_protein_indices)
            output["auroc"].append(auroc_val)
            output["auprc"].append(auprc_val)
            output["seed"].append(seed)

    # save the results
    output_df = pd.DataFrame(output)
    model_name = model_name if model_name is not None else "unknown_model"
    output_df.to_parquet(os.path.join(output_dir, f"operon_identification_results_{model_name}.parquet"))


class ArgParser(Tap):
    """Arguments for embedding DNA sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_df_filepath: str
    output_dir: str
    model_name: str
    n_negatives: int = 10


if __name__ == "__main__":
    args = ArgParser().parse_args()
    run(
        input_df_filepath=args.input_df_filepath,
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_negatives=args.n_negatives,
    )
