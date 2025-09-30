import os

import pandas as pd
import torch
from datasets import tqdm
from tap import Tap

from bacbench.modeling.embedder import load_seq_embedder
from bacbench.modeling.utils.utils_evo import preprocess_gene_seq_for_evo


def run(
    dataset_filepath: str,
    model_path: str,
    output_dir: str,
    start_idx: int = None,
    end_idx: int = None,
    max_seq_len: int = 8192,
):
    """
    Run the embedding of genes from the operon long read rna sequencing dataset.

    Args:
        dataset_name (str): Name of the dataset to process.
        model_path (str): Path to the model for embedding.
        output_dir (str): Directory to save the output files.
        start_idx (int | None): Start index for processing rows.
        end_idx (int | None): End index for processing rows.
        max_seq_len (int): Maximum sequence length for the model.
    """
    os.makedirs(output_dir, exist_ok=True)

    embedder = load_seq_embedder(model_path)

    df = pd.read_parquet(dataset_filepath)
    # select relevant rows for processing
    start_idx = start_idx if start_idx is not None else 0
    end_idx = end_idx if end_idx is not None else len(df)
    df = df.iloc[start_idx:end_idx]
    print(f"Processing {len(df)} rows, from {start_idx}, to {end_idx}.")

    df = df.explode(
        [
            "start",
            "end",
            "protein_sequence",
            "strand",
            "locus_tag",
            "old_locus_tag",
            "protein_name",
            "dna_sequence",
            "contig_len",
            "labels",
            "protein_names",
        ]
    )

    output = []
    for _, row in df.iterrows():
        # iterate through each gene in the row
        for start, end, strand in tqdm(zip(row["start"], row["end"], row["strand"], strict=False)):
            gene_seq, gene_mask = preprocess_gene_seq_for_evo(
                dna=row["dna_sequence"],
                start=start,
                end=end,
                strand=strand,
                max_seq_len=max_seq_len,
            )
            # embed the gene sequence using the Evo model
            with torch.no_grad():
                dna_representations = embedder([gene_seq], max_seq_len, pooling="mean", gene_mask=[gene_mask])
            output.append(
                {
                    "strain_name": row["strain_name"],
                    "contig_name": row["contig_name"],
                    "embeddings": dna_representations[0],
                }
            )

    output = pd.DataFrame(output).groupby(["strain_name", "contig_name"])[["embeddings"]].agg(list).reset_index()
    output = pd.merge(df, output, on=["strain_name", "contig_name"], how="inner")
    output = output.groupby("strain_name").agg(list).reset_index()

    output.to_parquet(
        os.path.join(output_dir, f"evo_{str(start_idx)}_{str(end_idx)}.parquet"),
        index=False,
    )


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # ──────────────────────────────────────────────────────────
    output_dir: str
    dataset_filepath: str = "/projects/u5ah/public/benchmarks/tasks/ppi/ppi_sample_combined.parquet"
    max_seq_len: int = 8192
    start_idx: int = None
    end_idx: int = None


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        dataset_filepath=args.dataset_filepath,
        model_path=args.model_path,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_seq_len=args.max_seq_len,
    )
