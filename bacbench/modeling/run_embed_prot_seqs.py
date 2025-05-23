import logging
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tap import Tap
from transformers import AutoModel

from bacbench.modeling.embed_prot_seqs import compute_bacformer_embeddings, compute_genome_protein_embeddings, load_plm
from bacbench.modeling.utils import get_prot_seq_col_name


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
    streaming: bool = False,
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
    :return: a pandas dataframe with the protein embeddings
    """
    # if dataset is a str, load dataset from HuggingFace
    if isinstance(dataset, str):
        dataset = load_dataset(dataset, streaming=streaming)

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
    for split_name, split_ds in dataset.items():
        prot_seq_col = get_prot_seq_col_name(split_ds.column_names)
        dataset = dataset.map(
            lambda row: add_protein_embeddings(
                row=row,
                prot_seq_col=prot_seq_col,  # noqa
                output_col=output_col,
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                batch_size=batch_size,
                max_prot_seq_len=max_prot_seq_len,
                genome_pooling_method=genome_pooling_method if model_type != "bacformer" else None,
            ),
            batched=False,
            keep_in_memory=False,
            remove_columns=[prot_seq_col],
        )
        if bacformer_model is not None:
            dataset = dataset.map(
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
                keep_in_memory=False,
            )
        # remove the protein sequence col as it takes up a lot of space and convert to pandas
        df = dataset.to_pandas()
        # add the split name
        df["split"] = split_name
        dfs.append(df)

    # concatenate splits
    dfs = pd.concat(dfs).reset_index(drop=True)
    return dfs


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    dataset_name: str
    streaming: bool = False
    output_filepath: str
    model_path: str
    model_type: Literal["esm2", "esmc", "protbert", "bacformer"]
    batch_size: int = 64
    max_prot_seq_len: int = 1024
    device: str = None
    output_col: str = "embeddings"
    genome_pooling_method: Literal["mean", "max"] = None
    max_n_proteins: int = 6000  # for Bacformer
    max_n_contigs: int = 1000  # for Bacformer


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    # load the dataset
    dataset = load_dataset(args.dataset_name, streaming=args.streaming)
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
    )
    # save the dataframe to parquet
    df.to_parquet(args.output_filepath)
