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
    max_seq_len: int = 8192,
):
    """
    Run the embedding of essential genes from a dataset.

    Args:
        dataset_name (str): Name of the dataset to process.
        model_path (str): Path to the model for embedding.
        output_dir (str): Directory to save the output files.
        save_every_n_rows (int): Save the output every n rows.
        max_seq_len (int): Maximum sequence length for the model.
    """
    os.makedirs(output_dir, exist_ok=True)

    embedder = load_seq_embedder(model_path)

    dataset = load_dataset(dataset_name)
    out_dfs = []
    for split_name, split_ds in dataset.items():
        df = split_ds.to_pandas()
        df["split"] = split_name
        out_dfs.append(df)
    df = pd.concat(out_dfs, ignore_index=True)
    df = df[["strain_name", "contig_name", "dna_sequence", "start", "end", "strand", "operon_prot_indices"]].explode(
        ["contig_name", "dna_sequence", "start", "end", "strand", "operon_prot_indices"]
    )

    output = []
    for _, row in df.iterrows():
        for start, end, strand in tqdm(zip(row["start"], row["end"], row["strand"], strict=False)):
            gene_seq, gene_mask = preprocess_gene_seq_for_evo(
                dna=row["dna_sequence"],
                start=start,
                end=end,
                strand=strand,
                max_seq_len=max_seq_len,
            )
            with torch.no_grad():
                dna_representations = embedder([gene_seq], max_seq_len, pooling="mean", gene_mask=[gene_mask])
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
        os.path.join(output_dir, "evo.parquet"),
        index=False,
    )


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # ──────────────────────────────────────────────────────────
    dataset_name: str = "macwiatrak/operon-identification-long-read-rna-sequencing-dna"
    model_path: str = "togethercomputer/evo-1-8k-base"
    max_seq_len: int = 8192
    output_dir: str = "/projects/u5ah/public/benchmarks/tasks/operon-long-read/"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
    )
