import os

import pandas as pd
import torch
from datasets import tqdm
from tap import Tap

from bacbench.modeling.embedder import load_seq_embedder

# from bacbench.modeling.utils.utils_glm2 import precompute_glm2_elements, preprocess_glm2_gene_seq
from bacbench.modeling.utils.scripts.utils_glm2 import precompute_glm2_elements, preprocess_glm2_gene_seq


def run(
    prot_parquet_path: str,
    dna_parquet_path: str,
    model_path: str,
    output_dir: str = None,
    output_filepath: str = None,
    save_every_n_rows: int = 1000,
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
    if output_filepath is not None and save_every_n_rows > 0:
        raise ValueError("Cannot specify both output_filepath and save_every_n_rows. Please choose one.")
    if output_filepath is None and output_dir is None:
        raise ValueError("Must specify either output_filepath or output_dir.")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    embedder = load_seq_embedder(model_path)

    prot_df = pd.read_parquet(prot_parquet_path)
    dna_df = pd.read_parquet(dna_parquet_path)
    prot_df = prot_df[
        ["genome_name", "contig_id", "start", "end", "strand", "essential", "protein_sequence", "split"]
    ].explode(["contig_id", "start", "end", "strand", "essential", "protein_sequence"])
    dna_df = dna_df[["genome_name", "contig_id", "dna_sequence"]].explode(["contig_id", "dna_sequence"])
    df = pd.merge(prot_df, dna_df, on=["genome_name", "contig_id"], how="inner")

    # merge the protein and DNA datasets on the genome name
    df = pd.merge(df, dna_df, on="genome_name", how="inner")

    output = []
    chunk_idx = 1
    # iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # precompute GLM2 elements for the gene sequences
        elements, gene_idx_to_elem_idx = precompute_glm2_elements(
            prot_seqs=row["protein_sequence"],
            dna_seq=row["dna_sequence"],
            start=row["start"],
            end=row["end"],
            strand=row["strand"],
        )
        # iterate through each gene in the row
        for gene_idx, (start, end, ess) in tqdm(
            enumerate(zip(row["start"], row["end"], row["essential"], strict=False))
        ):
            # preprocess the gene sequence for GLM2
            seq_str, gene_mask = preprocess_glm2_gene_seq(
                elements=elements,
                gene_idx_to_elem_idx=gene_idx_to_elem_idx,
                gene_idx=gene_idx,  # Assuming start is the gene index here
                max_seq_len=4096,
            )
            # embed the gene sequence
            with torch.no_grad():
                dna_representations = embedder([seq_str], max_seq_len, pooling="mean", gene_mask=[gene_mask])
            # prepare the output dictionary with relevant information
            output.append(
                {
                    "genome_name": row["genome_name"],
                    "contig_id": row["contig_id"],
                    "start": start,
                    "end": end,
                    "embeddings": dna_representations[0],
                    "split": row["split"],
                    "essential": ess,
                }
            )
            # save the output every `save_every_n_rows` rows
            if len(output) == save_every_n_rows:
                pd.DataFrame(output).to_parquet(
                    os.path.join(output_dir, f"chunk_{chunk_idx}_embeddings.parquet"),
                    index=False,
                )
                output = []
                chunk_idx += 1

    # save any remaining output
    if len(output) > 0 and save_every_n_rows > 0:
        pd.DataFrame(output).to_parquet(
            os.path.join(output_dir, f"chunk_{chunk_idx}_embeddings.parquet"),
            index=False,
        )
    if output_filepath is not None:
        pd.DataFrame(output).to_parquet(output_filepath, index=False)


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # ──────────────────────────────────────────────────────────
    prot_parquet_path: str
    dna_parquet_path: str
    output_dir: str = None  # output directory for saving the dataframe, only used for iterable datasets and if save_every_n_rows is set
    output_filepath: str = None  # output file path for saving the dataframe, only used if save_every_n_rows is not set
    model_path: str = "tattabio/gLM2_650M"
    max_seq_len: int = 4096
    save_every_n_rows: int = -1  # for saving the dataframe every n rows, only works for iterable datasets


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        prot_parquet_path=args.prot_parquet_path,
        dna_parquet_path=args.dna_parquet_path,
        output_filepath=args.output_filepath,
        model_path=args.model_path,
        output_dir=args.output_dir,
        save_every_n_rows=args.save_every_n_rows,
        max_seq_len=args.max_seq_len,
    )
