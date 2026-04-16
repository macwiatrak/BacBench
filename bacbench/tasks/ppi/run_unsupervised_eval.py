import json
import os

import pandas as pd
import torch
from datasets import tqdm
from tap import Tap
from torchmetrics.functional import auroc, average_precision


def run(
    input_filepath: str,
    train_test_split_filepath: list,
    output_dir: str,
    model_name: str,
    score_threshold: float = 0.6,
    max_n_proteins: int = 6000,
    max_n_ppi_pairs: float = 2 * 1e6,
):
    """Evaluate the model on the PPI dataset in an unsupervised manner."""
    # read in the dataset, only load necessary columns to save memory
    df = pd.read_parquet(input_filepath, columns=["strain_name", "labels", "embeddings"])

    # read in the train/test split
    with open(train_test_split_filepath) as f:
        split = json.load(f)

    # filter the dataset to only include test strains
    df["split"] = df["strain_name"].map(split)
    df = df[df["split"] == "test"]

    output = []
    for _, item in tqdm(df.iterrows()):
        genome_scores = []
        genome_labels = []
        for contig_labels, contig_embeddings in zip(item["labels"], item["embeddings"], strict=False):
            contig_labels = contig_labels[: int(max_n_ppi_pairs)]
            contig_embeddings = contig_embeddings[:max_n_proteins]
            for prot1_idx, prot2_idx, label in contig_labels:
                if prot1_idx >= len(contig_embeddings) or prot2_idx >= len(contig_embeddings):
                    continue
                # convert the label to be a value between 0 and 1 by dividing by 1000, this is because the original scores are between 0 and 1000
                label = label / 1000
                # binarize the labels based on the score threshold
                label = 1 if label >= score_threshold else 0
                # compute the cosine similarity between the two protein embeddings
                prot1_embedding = torch.tensor(contig_embeddings[prot1_idx]).unsqueeze(0)
                prot2_embedding = torch.tensor(contig_embeddings[prot2_idx]).unsqueeze(0)
                score = torch.cosine_similarity(prot1_embedding, prot2_embedding).item()
                genome_scores.append(score)
                genome_labels.append(label)

        # calculate genome-level AUROC and AUPRC
        try:
            genome_auroc = auroc(torch.tensor(genome_scores), torch.tensor(genome_labels), task="binary").item()
            genome_auprc = average_precision(
                torch.tensor(genome_scores), torch.tensor(genome_labels), task="binary"
            ).item()
            genome_metrics = {
                "strain_name": item["strain_name"],
                "auroc": genome_auroc,
                "auprc": genome_auprc,
                "n_ppi_pairs": len(genome_scores),
                "n_pos_ppi_pairs": sum(genome_labels),
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
    )
