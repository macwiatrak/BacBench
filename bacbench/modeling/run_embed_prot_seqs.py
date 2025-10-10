import logging
import os
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
import torch
from datasets import Dataset, IterableDataset, load_dataset
from tap import Tap
from transformers import AutoModel

try:
    from bacformer.modeling.modeling_updated import BacformerCGForMaskedGM
except ImportError:
    pass

from bacbench.modeling.embed_prot_seqs import compute_bacformer_embeddings, compute_genome_protein_embeddings
from bacbench.modeling.embedder import SeqEmbedder, load_seq_embedder
from bacbench.modeling.utils.utils import _iterable_to_dataframe, _slice_split, get_prot_seq_col_name


def add_protein_embeddings(
    row: dict[str, Any],
    prot_seq_col: str,
    output_col: str,
    embedder: SeqEmbedder,
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    genome_pooling_method: Literal["mean", "max"] = None,
):
    """Helper function to add protein embeddings to a row."""
    return {
        output_col: compute_genome_protein_embeddings(
            embedder=embedder,
            protein_sequences=row[prot_seq_col],
            contig_ids=row.get("contig_name", None),
            batch_size=batch_size,
            max_prot_seq_len=max_prot_seq_len,
            genome_pooling_method=genome_pooling_method,
        )
    }


def add_bacformer_embeddings(
    row: dict[str, Any],
    input_col: str,
    output_col: str,
    model: Callable,
    max_n_proteins: int = 6000,
    max_n_contigs: int = 1000,
    bacformer_model_type: Literal["base", "large"] = "base",
    genome_pooling_method: Literal["mean", "max"] = None,
) -> dict[str, Any]:
    """Helper function to add Bacformer embeddings to a row."""
    return {
        output_col: compute_bacformer_embeddings(
            model=model,
            protein_embeddings=row[input_col],
            contig_ids=row.get("contig_name", None),
            max_n_proteins=max_n_proteins,
            max_n_contigs=max_n_contigs,
            bacformer_model_type=bacformer_model_type,
            genome_pooling_method=genome_pooling_method,
        )
    }


def run(
    dataset: Dataset | None,
    model_path: str,
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    device: str = None,
    output_col: str = "embeddings",
    genome_pooling_method: Literal["mean", "max"] = None,
    max_n_proteins: int = 9000,  # for Bacformer
    max_n_contigs: int = 1000,  # for Bacformer
    start_idx: int | None = None,  # for slicing the dataset
    end_idx: int | None = None,  # for slicing the dataset
    save_every_n_rows: int | None = None,  # for saving the dataframe every n rows, only works for iterable datasets
    output_dir: str = None,  # output directory for saving the dataframe, only used for iterable datasets and if save_every_n_rows is set
):
    """Run script to embed protein sequences with various models.

    :param dataset: BacBench dataset
    :param model_path: model path to a HuggingFace model.
    :param batch_size: batch size for embedding pLMs
    :param max_prot_seq_len: max protein sequence length for embedding pLMs
    :param device: device to use for embedding pLMs, if None, will use cuda if available
    :param output_col: name of the output column for the embeddings
    :param genome_pooling_method: pooling method for the genome level embedding, one of ["mean", "max"]
    :param max_n_proteins: maximum number of proteins to use for each genome, only used for Bacformer
    :param max_n_contigs: maximum number of contigs to use for each genome, only used for Bacformer
    :param start_idx: start index for slicing the dataset, if None, will use the whole dataset
    :param end_idx: end index for slicing the dataset, if None, will use the whole dataset
    :param streaming: if True, will load the dataset in streaming mode, useful for large datasets
    :param save_every_n_rows: if set, will save the dataframe every n rows, only works for iterable datasets
    :param output_dir: output directory for saving the dataframe, only used for iterable datasets and if save_every_n_rows is set
    :return: a pandas dataframe with the protein embeddings
    """
    # set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # check if the model is Bacformer and adjust accordingly
    bacformer_model = None
    if "bacformer-300m" in model_path.lower():
        logging.info("Bacformer-300M model used, loading Bacformer-300M model and its ESM-C base model.")
        bacformer_model = (
            BacformerCGForMaskedGM.from_pretrained(model_path).bacformer.eval().to(torch.bfloat16).to(device)
        )
        model_path = "Synthyra/ESMplusplus_small"
        bacformer_model_type = "large"

    elif "bacformer" in model_path.lower():
        logging.info("Bacformer model used, loading Bacformer model and its ESM-2 base model.")
        bacformer_model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(torch.bfloat16).to(device)
        )
        model_path = "facebook/esm2_t12_35M_UR50D"
        bacformer_model_type = "base"

    # load pLM embedder
    embedder = load_seq_embedder(model_path)

    # embed protein sequences across splits
    dfs = []
    for split_name, split_ds in dataset.items():  # split_ds is a `Dataset`
        # slice the split
        split_ds = _slice_split(split_ds, start_idx, end_idx)
        # get the protein sequence column name
        prot_col = get_prot_seq_col_name(split_ds.column_names)

        # 1) embed every protein sequence in this split
        split_ds = split_ds.map(
            lambda row: add_protein_embeddings(
                row=row,
                prot_seq_col=prot_col,  # noqa
                output_col=output_col,
                embedder=embedder,
                batch_size=batch_size,
                max_prot_seq_len=max_prot_seq_len,
                genome_pooling_method=genome_pooling_method if bacformer_model is None else None,
            ),
            batched=False,
            remove_columns=[prot_col, "dna_sequence"] if "dna_sequence" in split_ds.column_names else [prot_col],
        )

        # 2) (optional) pass through Bacformer
        if bacformer_model is not None:
            split_ds = split_ds.map(
                lambda row: add_bacformer_embeddings(
                    row=row,
                    input_col=output_col,
                    output_col=output_col,
                    model=bacformer_model,
                    max_n_proteins=max_n_proteins,
                    max_n_contigs=max_n_contigs,
                    bacformer_model_type=bacformer_model_type,
                    genome_pooling_method=genome_pooling_method,
                ),
                batched=False,
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
    max_prot_seq_len: int = 1024
    device: str = None
    output_col: str = "embeddings"
    genome_pooling_method: Literal["mean", "max"] = None
    max_n_proteins: int = 9000  # for Bacformer
    max_n_contigs: int = 1000  # for Bacformer
    start_idx: int | None = None  # for slicing the dataset
    end_idx: int | None = None  # for slicing the dataset
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
        if "strain-clustering/prot-seqs-sample" in args.input_parquet_path:
            # read with fastparquet engine to avoid arrow issues
            dataset = Dataset.from_pandas(pd.read_parquet(data_files, engine="fastparquet"), preserve_index=False)
        else:
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
        max_prot_seq_len=args.max_prot_seq_len,
        device=args.device,
        output_col=args.output_col,
        genome_pooling_method=args.genome_pooling_method,
        max_n_proteins=args.max_n_proteins,
        max_n_contigs=args.max_n_contigs,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        save_every_n_rows=args.save_every_n_rows,
        output_dir=args.output_dir,
    )
    # save the dataframe if returned (i.e. if save_every_n_rows is not set)
    if df is not None:
        if args.output_filepath is not None:
            df.to_parquet(args.output_filepath)
        elif args.output_dir is not None:
            output_filepath = os.path.join(args.output_dir, f"chunk_{str(args.start_idx)}_{str(args.end_idx)}.parquet")
            df.to_parquet(output_filepath)
