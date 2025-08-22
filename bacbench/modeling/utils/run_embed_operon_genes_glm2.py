import os

import pandas as pd
import torch
from datasets import load_dataset, tqdm
from tap import Tap

from bacbench.modeling.embedder import load_seq_embedder
from bacbench.modeling.utils.utils_glm2 import precompute_glm2_elements, preprocess_glm2_gene_seq


def run(
    prot_dataset_name: str,
    dna_dataset_name: str,
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

    dataset = load_dataset(prot_dataset_name)
    out_dfs = []
    for split_name, split_ds in dataset.items():
        df = split_ds.to_pandas()
        df["split"] = split_name
        out_dfs.append(df)
    df = pd.concat(out_dfs, ignore_index=True)

    dataset = load_dataset(dna_dataset_name)
    dna_df = pd.concat([d.to_pandas() for d in dataset.values()], ignore_index=True)[["strain_name", "dna_sequence"]]

    df = pd.merge(df, dna_df, on="strain_name", how="inner")
    df = df[
        [
            "strain_name",
            "contig_name",
            "protein_sequence",
            "dna_sequence",
            "start",
            "end",
            "strand",
            "operon_prot_indices",
        ]
    ].explode(["contig_name", "protein_sequence", "dna_sequence", "start", "end", "strand", "operon_prot_indices"])

    output = []
    for _, row in df.iterrows():
        elements, gene_idx_to_elem_idx = precompute_glm2_elements(
            prot_seqs=row["protein_sequence"],
            dna_seq=row["dna_sequence"],
            start=row["start"],
            end=row["end"],
            strand=row["strand"],
        )

        for gene_idx in tqdm(enumerate(row["start"], strict=False)):
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
                    "contig_name": row["contig_name"],
                    "embeddings": dna_representations[0],
                }
            )

    output = pd.DataFrame(output).groupby(["contig_name"])[["embeddings"]].agg(list).reset_index()
    output = pd.merge(df, output, on="contig_name", how="inner")
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
    prot_dataset_name: str = "macwiatrak/operon-identification-long-read-rna-sequencing-protein-sequences"
    dna_dataset_name: str = "macwiatrak/operon-identification-long-read-rna-sequencing-dna"
    model_path: str = "tattabio/gLM2_650M"
    max_seq_len: int = 4096
    output_dir: str = "/projects/u5ah/public/benchmarks/tasks/operon-long-read/"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        prot_dataset_name=args.prot_dataset_name,
        dna_dataset_name=args.dna_dataset_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
    )
