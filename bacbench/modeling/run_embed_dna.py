import os
from typing import Any, Literal

import numpy as np
import pandas as pd
from datasets import Dataset, IterableDataset, load_dataset
from tap import Tap

from bacbench.modeling.embed_dna import embed_genome_dna_sequences
from bacbench.modeling.embedder import SeqEmbedder, load_seq_embedder
from bacbench.modeling.utils import _iterable_to_dataframe, _slice_split, get_dna_seq_col_name


def add_dna_embeddings(
    row: dict[str, Any],
    embedder: SeqEmbedder,
    dna_col: str,
    output_col: str,
    batch_size: int = 64,
    max_seq_len: int = 1024,
    dna_seq_overlap: int = 32,
    promoter_len: int = 128,
    genome_pooling_method: Literal["mean", "max"] = None,
    agg_whole_genome: bool = False,
) -> dict[str, list[np.ndarray] | np.ndarray]:
    """Embed genome DNA sequences using pretrained models.

    Args:
        row: dict[str, Any]: A dictionary representing a row from the dataset.
        embedder: The SeqEmbedder instance to use for embedding.
        dna_col (str): The column name containing the DNA sequence.
        output_col (str): The column name to store the embeddings.
        batch_size (int): The batch size to use for embedding.
        max_seq_len (int): The maximum sequence length for the model.
        dna_seq_overlap (int): The overlap between chunks of the DNA sequence.
        promoter_len (int): The length of the promoter region to include.
        genome_pooling_method (str): The pooling method to use for the genome level embedding.
            If None, list of DNA embedding chunks is returned.
        agg_whole_genome (bool): If True, the whole genome is embedded and aggregated.
    """
    # embed the dna sequence
    dna_seq = row[dna_col]
    if agg_whole_genome:
        if isinstance(dna_seq, list):
            # if the dna_col is a list, we assume it contains multiple contigs
            # and we concatenate them into a single string
            dna_seq = " ".join(dna_seq)
        embeddings = embed_genome_dna_sequences(
            embedder=embedder,
            dna=dna_seq,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dna_seq_overlap=dna_seq_overlap,
            promoter_len=promoter_len,
            genome_pooling_method=genome_pooling_method,
        )
    else:
        if isinstance(row["start"][0], list):
            dna_seq = dna_seq if isinstance(dna_seq, list) else dna_seq.split()
            embeddings = []
            for contig_dna_seq, start, end, strand in zip(
                dna_seq, row["start"], row["end"], row.get("strand", [None] * len(dna_seq)), strict=False
            ):
                embeddings.append(
                    embed_genome_dna_sequences(
                        embedder=embedder,
                        dna=contig_dna_seq,
                        start=start,
                        end=end,
                        strand=strand,
                        batch_size=batch_size,
                        max_seq_len=max_seq_len,
                        dna_seq_overlap=dna_seq_overlap,
                        promoter_len=promoter_len,
                        genome_pooling_method=genome_pooling_method,
                    )
                )
        else:
            embeddings = embed_genome_dna_sequences(
                embedder=embedder,
                dna=row[dna_col],
                start=row["start"],
                end=row["end"],
                strand=row["strand"],
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                dna_seq_overlap=dna_seq_overlap,
                promoter_len=promoter_len,
                genome_pooling_method=genome_pooling_method,
            )
    return {output_col: embeddings}


def run(
    dataset: Dataset | None,
    model_path: str,
    batch_size: int = 64,
    max_seq_len: int = 512,
    dna_seq_overlap: int = 32,
    promoter_len: int = 128,
    output_col: str = "embeddings",
    genome_pooling_method: Literal["mean", "max"] = None,
    agg_whole_genome: bool = False,
    start_idx: int | None = None,  # for slicing the dataset
    end_idx: int | None = None,  # for slicing the dataset
    save_every_n_rows: int | None = None,  # for saving the dataframe every n rows, only works for iterable datasets
    output_dir: str = None,  # output directory for saving the dataframe, only used for iterable datasets and if save_every_n_rows is set
):
    """Run script to embed DNA sequences with various models.

    :param dataset: Dataset to embed.
    :param model_path: Path to the model from HuggingFace.
    :param batch_size: Batch size for embedding.
    :param max_seq_len: Maximum sequence length for the model.
    :param dna_seq_overlap: Overlap between chunks of the DNA sequence.
    :param promoter_len: Length of the promoter region to include.
    :param output_col: Column name for the output embeddings.
    :param genome_pooling_method: Pooling method for the genome level embedding.
        If None, list of DNA embedding chunks is returned.
    :param agg_whole_genome: If True, the whole genome is embedded and aggregated.
        If False, the genome is at gene level. The former is used for genome-level tasks.
        The latter is used for gene-level tasks.
    :param start_idx: Start index for slicing the dataset.
    :param end_idx: End index for slicing the dataset.
    :param save_every_n_rows: if set, will save the dataframe every n rows, only works for iterable datasets
    :param output_dir: output directory for saving the dataframe, only used for iterable datasets and if save_every_n_rows is set
    :return: A pandas dataframe with the DNA embeddings.
    """
    # load DNA LM
    embedder = load_seq_embedder(model_path)

    # embed DNA sequences across splits
    dfs = []
    for split_name, split_ds in dataset.items():
        # slice the split
        split_ds = _slice_split(split_ds, start_idx, end_idx)
        # get the DNA sequence column name
        dna_col = get_dna_seq_col_name(split_ds.column_names)

        split_ds = split_ds.map(
            lambda row: add_dna_embeddings(
                row=row,
                embedder=embedder,
                dna_col=dna_col,  # noqa
                output_col=output_col,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                dna_seq_overlap=dna_seq_overlap,
                promoter_len=promoter_len,
                genome_pooling_method=genome_pooling_method,
                agg_whole_genome=agg_whole_genome,
            ),
            batched=False,
            remove_columns=[dna_col],
        )

        if isinstance(split_ds, IterableDataset):
            # Materialise – be cautious with very large datasets!
            df = _iterable_to_dataframe(
                split_ds,
                save_every_n_rows=save_every_n_rows,
                output_dir=output_dir,
                prefix=f"{split_name}_{start_idx}_{end_idx}_",
            )
        else:  # regular in-memory Dataset
            df = split_ds.to_pandas()

        # if save_every_n_rows is set, we already saved the dataframe in chunks
        if save_every_n_rows:
            continue

        df["split"] = split_name
        dfs.append(df)

    # if save_every_n_rows is set, we already saved the dataframe in chunks
    if save_every_n_rows:
        return

    # concatenate all splits and drop the index col we do not need
    df = pd.concat(dfs, ignore_index=True)
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])
    return df


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # use either dataset_name or parquet_file to load the dataset
    # ──────────────────────────────────────────────────────────
    dataset_name: str | None = None  # name of the HuggingFace dataset to load
    input_parquet_path: str | None = None  # path to a parquet file or dir to load and process
    # ──────────────────────────────────────────────────────────
    streaming: bool = False
    output_filepath: str = None
    model_path: str
    batch_size: int = 64
    max_seq_len: int = 512
    dna_seq_overlap: int = 32
    promoter_len: int = 128
    output_col: str = "embeddings"
    genome_pooling_method: Literal["mean", "max"] = None
    agg_whole_genome: bool = False
    start_idx: int | None = None
    end_idx: int | None = None
    save_every_n_rows: int = None  # for saving the dataframe every n rows, only works for iterable datasets
    output_dir: str = None  # output directory for saving the dataframe, only used for iterable datasets and if save_every_n_rows is set


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    # --  A) sanity-check input source  ---------------------------------
    if (args.dataset_name is None) == (args.input_parquet_path is None):
        raise ValueError("Provide **exactly one** of --dataset-name or --parquet-file.")
    # --  B) Load the dataset  ------------------------------------------
    if args.dataset_name:
        dataset = load_dataset(
            args.dataset_name,
            streaming=args.streaming,
            cache_dir=None,
        )
    else:  # parquet file chosen
        if os.path.isdir(args.input_parquet_path):
            data_files = [
                os.path.join(args.input_parquet_path, f)
                for f in os.listdir(args.input_parquet_path)
                if f.endswith(".parquet")
            ]
            # data_files = {"train": files}
            print(f"Loading {len(data_files)} parquet files from {args.input_parquet_path}")
        else:
            data_files = args.input_parquet_path
        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            streaming=args.streaming,
            cache_dir=None,
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # run the embedding
    df = run(
        dataset=dataset,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        dna_seq_overlap=args.dna_seq_overlap,
        promoter_len=args.promoter_len,
        output_col=args.output_col,
        genome_pooling_method=args.genome_pooling_method,
        agg_whole_genome=args.agg_whole_genome,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        save_every_n_rows=args.save_every_n_rows,
        output_dir=args.output_dir,
    )
    # if save_every_n_rows is set, we already saved the dataframe in chunks
    if args.save_every_n_rows is None:
        # save the dataframe to parquet
        df.to_parquet(args.output_filepath)
