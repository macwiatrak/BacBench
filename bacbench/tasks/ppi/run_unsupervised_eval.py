import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import tqdm
from tap import Tap
from torchmetrics.functional import auroc, average_precision


def _evaluate_contig_pairs(
    contig_labels,
    contig_embeddings,
    score_threshold: float,
    max_n_proteins: int,
    max_n_ppi_pairs: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Vectorize pair scoring for one contig."""
    contig_labels = contig_labels[: int(max_n_ppi_pairs)]
    contig_embeddings = contig_embeddings[:max_n_proteins]
    if len(contig_labels) == 0 or len(contig_embeddings) == 0:
        return None, None

    labels_array = np.asarray(contig_labels)
    if labels_array.dtype == np.object_:
        labels_array = np.stack([np.asarray(row, dtype=np.int64) for row in labels_array], axis=0)
    else:
        labels_array = labels_array.astype(np.int64, copy=False)
    if labels_array.size == 0:
        return None, None
    labels_tensor = torch.as_tensor(labels_array).reshape(-1, 3)

    embeddings_array = np.asarray(contig_embeddings)
    if embeddings_array.dtype == np.object_:
        embeddings_array = np.stack([np.asarray(row, dtype=np.float32) for row in embeddings_array], axis=0)
    else:
        embeddings_array = embeddings_array.astype(np.float32, copy=False)
    embeddings_tensor = torch.as_tensor(embeddings_array, dtype=torch.float32)
    n_embeddings = embeddings_tensor.shape[0]

    valid_mask = (labels_tensor[:, 0] < n_embeddings) & (labels_tensor[:, 1] < n_embeddings)
    if not torch.any(valid_mask):
        return None, None

    labels_tensor = labels_tensor[valid_mask]
    prot1_embeddings = embeddings_tensor[labels_tensor[:, 0].long()]
    prot2_embeddings = embeddings_tensor[labels_tensor[:, 1].long()]
    scores = F.cosine_similarity(prot1_embeddings, prot2_embeddings, dim=1)
    binary_labels = (labels_tensor[:, 2].to(torch.float32) / 1000.0 >= score_threshold).to(torch.long)
    return scores, binary_labels


def run(
    input_filepath: str,
    train_test_split_filepath: str,
    output_dir: str,
    model_name: str,
    score_threshold: float = 0.6,
    max_n_proteins: int = 6000,
    max_n_ppi_pairs: float = 2 * 1e6,
    embeddings_col: str = "embeddings",
):
    """Evaluate the model on the PPI dataset in an unsupervised manner."""
    # read in the dataset, only load necessary columns to save memory
    df = pd.read_parquet(input_filepath, columns=["strain_name", "labels", embeddings_col])

    # read in the train/test split
    with open(train_test_split_filepath) as f:
        split = json.load(f)

    # filter the dataset to only include test strains
    df["split"] = df["strain_name"].map(split)
    df = df[df["split"] == "test"]
    os.makedirs(output_dir, exist_ok=True)

    output = []
    for _, item in tqdm(df.iterrows(), total=len(df)):
        genome_scores = []
        genome_labels = []
        for contig_labels, contig_embeddings in zip(item["labels"], item[embeddings_col], strict=False):
            contig_scores, contig_binary_labels = _evaluate_contig_pairs(
                contig_labels=contig_labels,
                contig_embeddings=contig_embeddings,
                score_threshold=score_threshold,
                max_n_proteins=max_n_proteins,
                max_n_ppi_pairs=max_n_ppi_pairs,
            )
            if contig_scores is None or contig_binary_labels is None:
                continue
            genome_scores.append(contig_scores)
            genome_labels.append(contig_binary_labels)

        # calculate genome-level AUROC and AUPRC
        try:
            genome_scores_tensor = torch.cat(genome_scores)
            genome_labels_tensor = torch.cat(genome_labels)
            genome_auroc = auroc(genome_scores_tensor, genome_labels_tensor, task="binary").item()
            genome_auprc = average_precision(genome_scores_tensor, genome_labels_tensor, task="binary").item()
            genome_metrics = {
                "strain_name": item.strain_name,
                "auroc": genome_auroc,
                "auprc": genome_auprc,
                "n_ppi_pairs": int(genome_scores_tensor.numel()),
                "n_pos_ppi_pairs": int(genome_labels_tensor.sum().item()),
            }
            output.append(genome_metrics)
        except Exception as e:  # noqa
            print(f"Error in calculating genome-level metrics: {e}")
            continue

    if model_name is None:
        model_name = "unknown_model"

    output = pd.DataFrame(output)
    print(f"Mean AUROC: {output['auroc'].mean():.4f}, Mean AUPRC: {output['auprc'].mean():.4f}")
    print(f"Median AUROC: {output['auroc'].median():.4f}, Median AUPRC: {output['auprc'].median():.4f}")
    print(f"Std AUROC: {output['auroc'].std():.4f}, Std AUPRC: {output['auprc'].std():.4f}")

    output.to_csv(os.path.join(output_dir, f"unsupervised_eval_{model_name}.csv"), index=False)


class ArgumentParser(Tap):
    """Argument parser for training Bacformer (Lightning version)."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_filepath: str
    train_test_split_filepath: str
    output_dir: str
    model_name: str
    score_threshold: float = 0.6
    max_n_proteins: int = 6000
    max_n_ppi_pairs: float = 2 * 1e6
    embeddings_col: str = "embeddings"


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    args = parser.parse_args()
    run(
        input_filepath=args.input_filepath,
        train_test_split_filepath=args.train_test_split_filepath,
        output_dir=args.output_dir,
        model_name=args.model_name,
        score_threshold=args.score_threshold,
        max_n_proteins=args.max_n_proteins,
        max_n_ppi_pairs=args.max_n_ppi_pairs,
        embeddings_col=args.embeddings_col,
    )
