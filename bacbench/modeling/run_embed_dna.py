from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tap import Tap

from bacbench.modeling.embed_dna import embed_genome_dna_sequences, get_dna_seq, load_dna_lm
from bacbench.modeling.utils import get_dna_seq_col_name


def add_dna_embeddings(
    row: dict[str, Any],
    model: Callable,
    tokenizer: Callable,
    dna_col: str,
    output_col: str,
    model_type: Literal["nucleotide_transformer", "mistral_dna", "dnabert2"],
    batch_size: int = 64,
    max_seq_len: int = 1024,
    dna_seq_overlap: int = 32,
    promoter_len: int = 128,
    genome_pooling_method: Literal["mean", "max"] = None,
    agg_whole_genome: bool = False,
) -> dict[str, list[np.ndarray] | np.ndarray]:
    """Embed genome DNA sequences using pretrained models.

    Args:
        model (Callable): The pretrained model to use for embedding.
        tokenizer (Callable): The tokenizer to use for embedding.
        dna_sequence (str): The DNA sequence to embed.
        model_type (str): The type of model to use for embedding.
        batch_size (int): The batch size to use for embedding.
        max_seq_len (int): The maximum sequence length for the model.
        dna_seq_overlap (int): The overlap between chunks of the DNA sequence.
        promoter_len (int): The length of the promoter region to include.
        genome_pooling_method (str): The pooling method to use for the genome level embedding.
            If None, list of DNA embedding chunks is returned.
    """
    if agg_whole_genome:
        # embed the dna sequence
        embeddings = embed_genome_dna_sequences(
            model=model,
            tokenizer=tokenizer,
            dna_sequence=row[dna_col],
            model_type=model_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dna_seq_overlap=dna_seq_overlap,
            genome_pooling_method=genome_pooling_method,
        )
    else:
        embeddings = []
        for start, end, strand in zip(row["start"], row["end"], row["end"], strict=False):
            dna_seq = get_dna_seq(
                dna_seq=row[dna_col],
                start=int(start),
                end=int(end),
                strand=strand,
                promoter_len=promoter_len,
            )
            # row)[dna_col][int(start) : int(end)]
            # embed the dna sequence
            gene_embed = embed_genome_dna_sequences(
                model=model,
                tokenizer=tokenizer,
                dna_sequence=dna_seq,
                model_type=model_type,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                dna_seq_overlap=dna_seq_overlap,
                genome_pooling_method=genome_pooling_method,
            )
            embeddings.append(gene_embed)

    return {output_col: embeddings}


def run(
    dataset: Dataset | None,
    model_path: str,
    model_type: Literal["nucleotide_transformer", "mistral_dna", "dnabert2"],
    batch_size: int = 64,
    max_seq_len: int = 512,
    dna_seq_overlap: int = 32,
    promoter_len: int = 128,
    output_col: str = "embeddings",
    genome_pooling_method: Literal["mean", "max"] = None,
    agg_whole_genome: bool = False,
    streaming: bool = False,
):
    """Run script to embed DNA sequences with various models.

    :param dataset: Dataset to embed.
    :param model_path: Path to the model.
    :param model_type: Type of the model to use.
    :param batch_size: Batch size for embedding.
    :param max_seq_len: Maximum sequence length for the model.
    :param dna_seq_overlap: Overlap between chunks of the DNA sequence.
    :param device: Device to use for embedding.
    :param output_col: Column name for the output embeddings.
    :param genome_pooling_method: Pooling method for the genome level embedding.
        If None, list of DNA embedding chunks is returned.
    :param agg_whole_genome: If True, the whole genome is embedded and aggregated.
        If False, the genome is at gene level. The former is used for genome-level tasks.
        The latter is used for gene-level tasks.
    :return: A pandas dataframe with the DNA embeddings.
    """
    # if dataset is a str, load dataset from HuggingFace
    if isinstance(dataset, str):
        dataset = load_dataset(dataset, streaming=streaming)

    # load DNA LM
    model, tokenizer = load_dna_lm(model_path, model_type)

    # embed DNA sequences across splits
    dfs = []
    for split_name, split_ds in dataset.items():
        dna_col = get_dna_seq_col_name(split_ds.column_names)

        split_ds = dataset.map(
            lambda row: add_dna_embeddings(
                row=row,
                model=model,
                tokenizer=tokenizer,
                dna_col=dna_col,  # noqa
                output_col=output_col,
                model_type=model_type,
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

        # 3) convert THIS split to pandas
        df = split_ds.to_pandas()
        df["split"] = split_name
        dfs.append(df)

    # concatenate all splits
    df = pd.concat(dfs, ignore_index=True)
    return df


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    dataset_name: str
    streaming: bool = False
    output_filepath: str
    model_path: str
    model_type: Literal["nucleotide_transformer", "mistral_dna", "dnabert2"]
    batch_size: int = 64
    max_seq_len: int = 512
    dna_seq_overlap: int = 32
    promoter_len: int = 128
    output_col: str = "embeddings"
    genome_pooling_method: Literal["mean", "max"] = None
    agg_whole_genome: bool = False


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
        max_seq_len=args.max_seq_len,
        dna_seq_overlap=args.dna_seq_overlap,
        promoter_len=args.promoter_len,
        output_col=args.output_col,
        genome_pooling_method=args.genome_pooling_method,
        agg_whole_genome=args.agg_whole_genome,
        streaming=args.streaming,
    )
    # save the dataframe to parquet
    df.to_parquet(args.output_filepath)
