import logging
import os
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
import torch
from datasets import Dataset, IterableDataset, load_dataset
from tap import Tap
from transformers import AutoModel

from bacbench.modeling.embed_prot_seqs import compute_bacformer_embeddings, compute_genome_protein_embeddings, load_plm
from bacbench.modeling.utils import _iterable_to_dataframe, _slice_split, get_prot_seq_col_name


def add_protein_embeddings(
    row: dict[str, Any],
    prot_seq_col: str,
    output_col: str,
    model: Callable,
    tokenizer: Callable,
    model_type: Literal["esm2", "esmc", "protbert"] = "esm2",
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    genome_pooling_method: Literal["mean", "max"] = None,
):
    """Helper function to add protein embeddings to a row."""
    return {
        output_col: compute_genome_protein_embeddings(
            model=model,
            tokenizer=tokenizer,
            protein_sequences=row[prot_seq_col],
            contig_ids=row.get("contig_name", None),
            model_type=model_type,
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
            genome_pooling_method=genome_pooling_method,
        )
    }


def run(
    dataset: Dataset | None,
    model_path: str,
    model_type: Literal["esm2", "esmc", "protbert", "bacformer"],
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    device: str = None,
    output_col: str = "embeddings",
    genome_pooling_method: Literal["mean", "max"] = None,
    max_n_proteins: int = 6000,  # for Bacformer
    max_n_contigs: int = 1000,  # for Bacformer
    start_idx: int | None = None,  # for slicing the dataset
    end_idx: int | None = None,  # for slicing the dataset
    streaming: bool = False,
    save_every_n_rows: int | None = None,  # for saving the dataframe every n rows, only works for iterable datasets
    output_dir: str = None,  # output directory for saving the dataframe, only used for iterable datasets and if save_every_n_rows is set
):
    """Run script to embed protein sequences with various models.

    :param dataset: BacBench dataset
    :param model_path: sHuggingFace model name or path to model
    :param model_type: the used embedding model one of ["esm2", "esmc", "protbert", "bacformer"]
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
    if model_type == "bacformer":
        logging.info("Bacformer model used, loading Bacformer model and its ESM-2 base model.")
        bacformer_model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(torch.bfloat16).to(device)
        )
        model_type = "esm2"
        model_path = "facebook/esm2_t12_35M_UR50D"

    # load pLM
    model, tokenizer = load_plm(model_path, model_type)

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
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                batch_size=batch_size,
                max_prot_seq_len=max_prot_seq_len,
                genome_pooling_method=genome_pooling_method if bacformer_model is None else None,
            ),
            batched=False,
            remove_columns=[prot_col],
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
                prefix=f"{split_name}_",
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
    input_parquet_filepath: str | None = None  # path to a parquet file to load and process
    # ──────────────────────────────────────────────────────────
    streaming: bool = False
    output_filepath: str = None
    model_path: str
    model_type: Literal["esm2", "esmc", "protbert", "bacformer"]
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
    if (args.dataset_name is None) == (args.input_parquet_filepath is None):
        raise ValueError("Provide **exactly one** of --dataset-name or --parquet-file.")
    # --  B) Load the dataset  ------------------------------------------
    if args.dataset_name:
        dataset = load_dataset(
            args.dataset_name,
            streaming=args.streaming,
            cache_dir=None,
        )
    else:  # parquet file chosen
        dataset = load_dataset(
            "parquet",
            data_files=args.input_parquet_filepath,
            streaming=args.streaming,
            cache_dir=None,
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # run the embedding
    df = run(
        dataset=dataset,
        model_path=args.model_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        max_prot_seq_len=args.max_prot_seq_len,
        device=args.device,
        output_col=args.output_col,
        genome_pooling_method=args.genome_pooling_method,
        max_n_proteins=args.max_n_proteins,
        max_n_contigs=args.max_n_contigs,
        streaming=args.streaming,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        save_every_n_rows=args.save_every_n_rows,
        output_dir=args.output_dir,
    )
    # if save_every_n_rows is set, we already saved the dataframe in chunks
    if args.save_every_n_rows is None:
        # save the dataframe to parquet
        df.to_parquet(args.output_filepath)
