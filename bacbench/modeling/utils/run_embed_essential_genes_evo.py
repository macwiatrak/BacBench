import os

import pandas as pd
import torch
from datasets import load_dataset, tqdm
from tap import Tap

from bacbench.modeling.embedder import load_seq_embedder
from bacbench.modeling.utils.utils_evo import preprocess_gene_seq_for_evo


def run(
    dataset_name: str,
    model_path: str,
    output_dir: str,
    save_every_n_rows: int = 1000,
    max_seq_len: int = 8192,
    start_idx: int | None = None,
    end_idx: int | None = None,
):
    """
    Run the embedding of essential genes from a datase with the Evo model.

    Args:
        dataset_name (str): Name of the dataset to process.
        model_path (str): Path to the model for embedding.
        output_dir (str): Directory to save the output files.
        save_every_n_rows (int): Save the output every n rows.
        max_seq_len (int): Maximum sequence length for the model.
        start_idx (int | None): Start index for processing rows.
        end_idx (int | None): End index for processing rows.
    """
    os.makedirs(output_dir, exist_ok=True)

    embedder = load_seq_embedder(model_path)

    dataset = load_dataset(dataset_name)
    out_dfs = []
    # convert each split of the dataset to a pandas DataFrame and add a 'split' column
    for split_name, split_ds in dataset.items():
        df = split_ds.to_pandas()
        df["split"] = split_name
        out_dfs.append(df)
    df = pd.concat(out_dfs, ignore_index=True)

    # select relevant rows for processing
    start_idx = start_idx if start_idx is not None else 0
    end_idx = end_idx if end_idx is not None else len(df)
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    print(f"Processing {len(df)} rows, from {start_idx}, to {end_idx}.")

    output = []
    chunk_idx = 1
    # iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # iterate through each gene
        for start, end, strand, ess in tqdm(
            zip(row["start"], row["end"], row["strand"], row["essential"], strict=False)
        ):
            # preprocess the gene sequence for Evo model, adding context from both sides
            gene_seq, gene_mask = preprocess_gene_seq_for_evo(
                dna=row["dna_seq"],
                start=start,
                end=end,
                strand=strand,
                max_seq_len=max_seq_len,
            )
            # embed the gene sequence using the Evo model
            with torch.no_grad():
                dna_representations = embedder([gene_seq], max_seq_len, pooling="mean", gene_mask=[gene_mask])
            # prepare the output dictionary with relevant information
            output.append(
                {
                    "genome_name": row["genome_name"],
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "embeddings": dna_representations[0],
                    "split": row["split"],
                    "essential": ess,
                }
            )
            # save the output every `save_every_n_rows` rows
            if len(output) == save_every_n_rows:
                pd.DataFrame(output).to_parquet(
                    os.path.join(output_dir, f"chunk_{chunk_idx}_start_{start_idx}_end_{end_idx}_embeddings.parquet"),
                    index=False,
                )
                output = []
                chunk_idx += 1
    # save any remaining output
    if len(output) > 0:
        pd.DataFrame(output).to_parquet(
            os.path.join(output_dir, f"chunk_{chunk_idx}_start_{start_idx}_end_{end_idx}_embeddings.parquet"),
            index=False,
        )


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # ──────────────────────────────────────────────────────────
    output_dir: str
    dataset_name: str = "macwiatrak/bacbench-essential-genes-dna"
    model_path: str = "togethercomputer/evo-1-8k-base"
    max_seq_len: int = 8192
    start_idx: int | None = None
    end_idx: int | None = None
    save_every_n_rows: int = 10000  # for saving the dataframe every n rows, only works for iterable datasets


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        save_every_n_rows=args.save_every_n_rows,
        max_seq_len=args.max_seq_len,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
