import os

import pandas as pd
import torch
from datasets import tqdm
from tap import Tap
from torchmetrics.functional import auroc, average_precision

from bacbench.tasks.ppi.data_reader import process_triples


def eval_cs_ppi(protein_embeddings: torch.Tensor, triples: torch.Tensor):
    """Evaluate the model on the PPI dataset in an unsupervised manner."""
    prot1 = protein_embeddings[triples[0, :]]
    prot2 = protein_embeddings[triples[1, :]]
    scores = torch.cosine_similarity(prot1, prot2)
    auroc_val = auroc(scores, triples[2, :], task="binary").item()
    auprc_val = average_precision(scores, triples[2, :], task="binary").item()
    return auroc_val, auprc_val


def run(
    input_dir: str,
    output_dir: str,
    model_name: str,
    score_threshold: float = 0.6,
    max_n_proteins: int = 6000,
    max_n_ppi_pairs: float = 2 * 1e6,
):
    """Evaluate the model on the PPI dataset in an unsupervised manner."""
    test_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith("parquet")]
    print(f"Found {len(test_files)} files in {input_dir}.")

    output = []
    for _, f in tqdm(enumerate(test_files[1:])):
        df = pd.read_parquet(f)
        for idx, item in tqdm(df.iterrows()):
            triples = process_triples(
                triples=item["triples_combined_score"],
                score_threshold=score_threshold,
                max_n_proteins=min(len(item["protein_embeddings"]), max_n_proteins),
                max_n_ppi_pairs=max_n_ppi_pairs,
            )
            protein_embeddings = torch.stack([torch.tensor(i) for i in item["protein_embeddings"]])
            try:
                auroc, auprc = eval_cs_ppi(protein_embeddings, triples)
            except Exception as e:  # noqa
                print(f"Error in eval_cs_ppi: {e}")
                print(f"Skipping file {os.path.basename(f)} with index {idx} due to error.")
                continue
            output.append(
                {
                    "genome_name": item["genome_name"],
                    "auroc": auroc,
                    "auprc": auprc,
                    "n_proteins": protein_embeddings.shape[0],
                    "n_ppi_pairs": triples.shape[1],
                    "n_pos_triples": (triples[2, :] == 1).sum().item(),
                }
            )
        pd.DataFrame(output).to_csv(os.path.join(output_dir, f"unsupervised_eval_{model_name}.csv"), index=False)

    output = pd.DataFrame(output)
    output.to_csv(os.path.join(output_dir, f"unsupervised_eval_{model_name}.csv"), index=False)


class ArgumentParser(Tap):
    """Argument parser for training Bacformer (Lightning version)."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str
    output_dir: str
    model_name: str = "esmc"
    score_threshold: float = 0.6
    max_n_proteins: int = 6000
    max_n_ppi_pairs: float = 2 * 1e6


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    args = parser.parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        score_threshold=args.score_threshold,
        max_n_proteins=args.max_n_proteins,
        max_n_ppi_pairs=args.max_n_ppi_pairs,
    )
