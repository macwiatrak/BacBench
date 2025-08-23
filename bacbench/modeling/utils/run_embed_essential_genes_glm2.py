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
    os.makedirs(output_dir, exist_ok=True)

    embedder = load_seq_embedder(model_path)

    dataset = load_dataset(prot_dataset_name)
    out_dfs = []
    # convert each split of the dataset to a pandas DataFrame and add a 'split' column
    for split_name, split_ds in dataset.items():
        df = split_ds.to_pandas()
        df["split"] = split_name
        out_dfs.append(df)
    df = pd.concat(out_dfs, ignore_index=True)

    # select relevant rows for processing
    dataset = load_dataset(dna_dataset_name)
    # concatenate all splits of the DNA dataset into a single DataFrame
    dna_df = pd.concat([d.to_pandas() for d in dataset.values()], ignore_index=True)[
        ["genome_name", "dna_seq", "strand"]
    ]

    # merge the protein and DNA datasets on the genome name
    df = pd.merge(df, dna_df, on="genome_name", how="inner")

    output = []
    chunk_idx = 1
    # iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # precompute GLM2 elements for the gene sequences
        elements, gene_idx_to_elem_idx = precompute_glm2_elements(
            prot_seqs=row["sequence"],
            dna_seq=row["dna_seq"],
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
                    "contig_name": row["contig_name"],
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
    if len(output) > 0:
        pd.DataFrame(output).to_parquet(
            os.path.join(output_dir, f"chunk_{chunk_idx}_embeddings.parquet"),
            index=False,
        )


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # ──────────────────────────────────────────────────────────
    output_dir: str  # output directory for saving the dataframe, only used for iterable datasets and if save_every_n_rows is set
    prot_dataset_name: str = "macwiatrak/bacbench-essential-genes-protein-sequences"
    dna_dataset_name: str = "macwiatrak/bacbench-essential-genes-dna"
    model_path: str = "tattabio/gLM2_650M"
    max_seq_len: int = 4096
    save_every_n_rows: int = 20000  # for saving the dataframe every n rows, only works for iterable datasets


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        prot_dataset_name=args.prot_dataset_name,
        dna_dataset_name=args.dna_dataset_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        save_every_n_rows=args.save_every_n_rows,
        max_seq_len=args.max_seq_len,
    )
