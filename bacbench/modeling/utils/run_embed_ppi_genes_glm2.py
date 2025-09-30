import os

import pandas as pd
import torch
from datasets import tqdm
from tap import Tap

from bacbench.modeling.embedder import load_seq_embedder
from bacbench.modeling.utils.utils_glm2 import precompute_glm2_elements, preprocess_glm2_gene_seq


def run(
    dataset_filepath: str,
    model_path: str,
    output_dir: str,
    max_seq_len: int = 4096,
):
    """
    Run the embedding of essential genes from a dataset.

    Args:
        prot_dataset_name (str): Name of the protein dataset to process.
        dna_dataset_name (str): Name of the DNA dataset to process.
        model_path (str): Path to the model for embedding.
        output_dir (str): Directory to save the output files.
        save_every_n_rows (int): Save the output every n rows.
        max_seq_len (int): Maximum sequence length for the model.
    """
    os.makedirs(output_dir, exist_ok=True)

    embedder = load_seq_embedder(model_path)

    df = pd.read_parquet(dataset_filepath)
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
        elements, gene_idx_to_elem_idx = precompute_glm2_elements(
            prot_seqs=row["protein_sequence"],
            dna_seq=row["dna_sequence"],
            start=row["start"],
            end=row["end"],
            strand=row["strand"],
        )

        for gene_idx in tqdm(range(len(row["start"]))):
            seq_str, gene_mask = preprocess_glm2_gene_seq(
                elements=elements,
                gene_idx_to_elem_idx=gene_idx_to_elem_idx,
                gene_idx=gene_idx,  # Assuming start is the gene index here
                max_seq_len=4096,
            )
            with torch.no_grad():
                dna_representations = embedder([seq_str], max_seq_len, pooling="mean", gene_mask=[gene_mask])
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
        os.path.join(output_dir, "glm2.parquet"),
        index=False,
    )


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # ──────────────────────────────────────────────────────────
    output_dir: str
    dataset_filepath: str = "/projects/u5ah/public/benchmarks/tasks/ppi/ppi_sample_combined.parquet"
    model_path: str = "tattabio/gLM2_650M"
    max_seq_len: int = 4096


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        dataset_filepath=args.dataset_filepath,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
    )
