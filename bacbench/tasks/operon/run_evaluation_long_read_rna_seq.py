import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tap import Tap


def operon_prot_indices_to_pairwise_labels(operon_prot_indices: list[list[int]], n_genes: int) -> np.ndarray:
    """Convert operon protein indices to pairwise binary labels across whole sequence.

    Parameters
    ----------
    operon_prot_indices : list of operons; each operon is a list of 0-based gene indices
    n_genes  : total number of genes (N)

    Returns
    -------
    np.ndarray  (shape = (N-1,))
        binary[i] = 1  ⇔  genes i and i+1 are in the SAME operon
    """
    binary = np.zeros(n_genes - 1, dtype=int)

    for operon in operon_prot_indices:
        max_operon = max(operon)
        min_operon = min(operon)
        # mark every adjacent pair inside that operon
        for item in range(min_operon, max_operon):
            binary[item] = 1

    return binary


def get_intergenic_bp_dist(starts: list[int], ends: list[int]) -> np.ndarray:
    """Compute intergenic distances in base pairs between genes."""
    out = []
    for idx in range(len(starts) - 1):
        d = starts[idx + 1] - ends[idx]
        out.append(d)
    return np.array(out)


def predict_pairwise_operon_boundaries(
    emb: np.ndarray,
    intergenic_bp: np.ndarray,
    strand: np.ndarray,
    scale_bp: int = 500,
    max_gap: int = 500,
) -> np.ndarray:
    """Predict pairwise operon boundaries based on embeddings and intergenic distances.

    params
    :param emb: Embeddings of the (avg) protein sequences, shape (n, d).
    :param intergenic_bp: Intergenic distances in base pairs, shape (n-1,).
    :param strand: Strand information, shape (n,).
    :param scale_bp: Scaling factor for intergenic distances, default 500.
    :param max_gap: Maximum gap allowed for operon prediction, default 500.

    :return: predicted operon boundary scores
    """
    emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = np.sum(emb_n[:-1] * emb_n[1:], axis=1).reshape(-1, 1)
    cos = (cos + 1) / 2

    same = (strand[:-1] == strand[1:]).astype(int)
    diff_mask = same == 0

    # exponential distance weighting
    if scale_bp is not None:
        cos[:, 0] *= np.exp(-intergenic_bp / scale_bp)

    cos[diff_mask] = 0.0  # strand veto

    # ---------- NEW distance-gate veto ------------------------------------
    if max_gap is not None:
        cos[intergenic_bp > max_gap] = 0.0  # hard boundary

    return cos[:, 0]


def run(
    input_filepath: str,
    output_filepath: str,
):
    """Run the operon identification evaluation using long-read RNA seq data"""
    df = pd.read_parquet(input_filepath)

    # explode the dataset by contig, this allows prediction per contig
    df = df.explode(
        [
            "contig_name",
            "gene_name",
            "locus_tag",
            "start",
            "end",
            "strand",
            "protein_id",
            "embeddings",
            "operon_prot_indices",
        ]
    )

    # compute the intergenic distances
    df["intergenic_bp"] = df.apply(
        lambda row: get_intergenic_bp_dist(
            starts=row["start"],
            ends=row["end"],
        ),
        axis=1,
    )

    # get the labels
    df["operon_pairwise_labels"] = df.apply(
        lambda row: operon_prot_indices_to_pairwise_labels(
            operon_prot_indices=row["operon_prot_indices"],
            n_genes=len(row["embeddings"]),
        ),
        axis=1,
    )

    # run the operon prediction
    df["operon_pairwise_scores"] = df.apply(
        lambda row: predict_pairwise_operon_boundaries(
            emb=np.stack(row["embeddings"]),
            intergenic_bp=row["intergenic_bp"],
            strand=row["strand"],
            scale_bp=500,
            max_gap=500,
        ),
        axis=1,
    )

    # group contigs by strain and aggregate the results
    df = df.groupby("strain_name")[["operon_pairwise_labels", "operon_pairwise_scores"]].agg(list)
    df["operon_pairwise_labels"] = df["operon_pairwise_labels"].apply(lambda x: list(itertools.chain(*x)))
    df["operon_pairwise_scores"] = df["operon_pairwise_scores"].apply(lambda x: list(itertools.chain(*x)))

    # compute metrics
    df["AUROC"] = df.apply(lambda row: roc_auc_score(row.operon_pairwise_labels, row.operon_pairwise_scores), axis=1)
    df["AUPRC"] = df.apply(
        lambda row: average_precision_score(row.operon_pairwise_labels, row.operon_pairwise_scores), axis=1
    )

    print(f"Mean AUROC: {df['AUROC'].mean():.4f}, Mean AUPRC: {df['AUPRC'].mean():.4f}")
    print(f"Std AUROC: {df['AUROC'].std():.4f}, Std AUPRC: {df['AUPRC'].std():.4f}")

    # save the data
    if output_filepath is not None:
        df.to_csv(output_filepath)


class ArgumentParser(Tap):
    """Argument parser for finetuning linear model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_filepath: str = "/Users/maciejwiatrak/Downloads/mistral.parquet"
    output_filepath: str = None


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(input_filepath=args.input_filepath, output_filepath=args.output_filepath)
