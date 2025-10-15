from typing import Literal

import pandas as pd
import torch
from calflops import calculate_flops
from tap import Tap
from torch import nn
from tqdm import tqdm

from bacbench.modeling.embed_dna import chunk_whole_genome_dna_seq
from bacbench.modeling.embedder import SeqEmbedder, load_seq_embedder


def calculate_genome_flops(
    input_seqs: list[str],  # protein or DNA sequences
    embedder: SeqEmbedder,
    max_seq_len: int,
    bp_factor: float = 2.0,  # backprop factor (2.0 for Adam, 1.0 for SGD)
):
    """Calculate flops for a model on a list of protein or DNA sequences."""
    # calculate input_seqs lens
    output = []
    for seq in tqdm(input_seqs):
        seqs = embedder._preprocess_seqs([seq])
        inputs = embedder._tokenize(seqs, max_seq_len=max_seq_len)
        with torch.no_grad():
            fwd_flops, fwd_macs, params = calculate_flops(
                model=embedder.model,
                kwargs=inputs,  # your inputs dict
                include_backPropagation=False,  # forward only
                print_results=False,
                print_detailed=False,
                output_as_string=False,
            )
        train_flops = fwd_flops * (1.0 + bp_factor)
        train_macs = fwd_macs * (1.0 + bp_factor)
        output.append(
            {
                "n_training_params": params,
                "fwd_MACs": fwd_macs,
                "fwd_FLOPs": fwd_flops,
                "fwd+bwd_MACs": train_macs,
                "fwd+bwd_FLOPs": train_flops,
            }
        )
    output = pd.DataFrame(output)
    # extract statistics
    overall_stats = {
        "n_params": output["n_training_params"].iloc[0],
        "total_flops_fwd": output["fwd_FLOPs"].sum(),
        "total_flops_fwd+bwd": output["fwd+bwd_FLOPs"].sum(),
        "total_macs_fwd": output["fwd_MACs"].sum(),
        "total_macs_fwd+bwd": output["fwd+bwd_MACs"].sum(),
    }
    return overall_stats


def calculate_genome_bacformer(
    input_seqs: list[str],  # protein or DNA sequences
    model: nn.Module,
    max_n_proteins: int = 9000,
    bp_factor: float = 2.0,  # backprop factor (2.0 for Adam, 1.0 for SGD)
    bacformer_model_type: Literal["base", "large"] = "base",
):
    """Calculate flops for Bacformer model on a list of protein sequences."""
    # calculate input_seqs lens
    n_proteins = min(len(input_seqs), max_n_proteins)
    if bacformer_model_type == "base":
        embed_dim = 480
        inputs = {
            "protein_embeddings": torch.randn(1, n_proteins, embed_dim).type(torch.bfloat16),
            "attention_mask": torch.ones(1, n_proteins),
            "token_type_ids": torch.zeros(1, n_proteins, dtype=torch.long),
            "special_tokens_mask": torch.ones(1, n_proteins, dtype=torch.long) * 4,
        }
    else:
        embed_dim = 960
        inputs = {
            "protein_embeddings": torch.randn(1, n_proteins, embed_dim).type(torch.bfloat16),
            "attention_mask": torch.ones(1, n_proteins),
            "contig_ids": torch.zeros(1, n_proteins, dtype=torch.long),
        }
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        fwd_flops, fwd_macs, params = calculate_flops(
            model=model,
            kwargs=inputs,  # your inputs dict
            include_backPropagation=False,  # forward only
            print_results=False,
            print_detailed=False,
            output_as_string=False,
        )
    train_flops = fwd_flops * (1.0 + bp_factor)
    train_macs = fwd_macs * (1.0 + bp_factor)
    # extract statistics
    overall_stats = {
        "n_params": params,
        "total_flops_fwd": fwd_flops,
        "total_flops_fwd+bwd": train_flops,
        "total_macs_fwd": fwd_macs,
        "total_macs_fwd+bwd": train_macs,
    }
    return overall_stats


def run(
    input_df_path: str,
    model_name_or_path: str,
    max_seq_len: int,
    max_n_proteins: int,
    model_type: Literal["dna", "protein", "bacformer", "glm2"],
    output_filepath: str = None,
    dna_seq_overlap: int = None,
):
    """Run flops calculation on a full genome."""
    embedder = load_seq_embedder(model_name_or_path)

    # load the data
    df = pd.read_parquet(input_df_path)
    out = []
    for _, row in df.iterrows():
        if model_type in ["protein", "bacformer"]:
            input_seqs = row["protein_sequences"]
        elif model_type == "dna":
            input_seqs = " ".join(row["dna_sequence"])
            input_seqs = chunk_whole_genome_dna_seq(
                dna_sequence=input_seqs, max_seq_len=max_seq_len, overlap=dna_seq_overlap
            )
        else:
            # process the data into gLM2 format
            input_seqs = " ".join(df["dna_sequence"])
            input_seqs = chunk_whole_genome_dna_seq(
                dna_sequence=input_seqs, max_seq_len=max_seq_len, overlap=dna_seq_overlap
            )

        stats = calculate_genome_flops(
            input_seqs=input_seqs,
            embedder=embedder,
            max_seq_len=max_seq_len,
        )
        if model_type == "bacformer":
            bacformer_stats = calculate_genome_bacformer(
                input_seqs=input_seqs,
                model=embedder.model,
                max_n_proteins=max_n_proteins,
                bacformer_model_type="large" if "bacformer-300m" in model_name_or_path.lower() else "base",
            )
            stats["total_flops_fwd"] += bacformer_stats["total_flops_fwd"]
            stats["total_flops_fwd+bwd"] += bacformer_stats["total_flops_fwd+bwd"]
            stats["total_macs_fwd"] += bacformer_stats["total_macs_fwd"]
            stats["total_macs_fwd+bwd"] += bacformer_stats["total_macs_fwd+bwd"]
            stats["n_params"] = bacformer_stats["n_params"]
        print(f"{model_name_or_path} on {row['strain_name']} statistics:", stats)
        stats["strain_name"] = row["strain_name"]
        out.append(stats)

    if output_filepath is not None:
        pd.DataFrame(out).to_parquet(output_filepath)


class ArgumentParser(Tap):
    """Argument parser for running flop calculation on whole genomes."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_df_path: str
    model_name_or_path: str
    model_type: Literal["dna", "protein", "bacformer", "glm2"]
    max_seq_len: int
    max_n_proteins: int = 9000  # only for Bacformer models
    output_filepath: str = None
    dna_seq_overlap: int = None  # only for DNA models


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_df_path=args.input_df_path,
        model_name_or_path=args.model_name_or_path,
        max_seq_len=args.max_seq_len,
        max_n_proteins=args.max_n_proteins,
        model_type=args.model_type,
        output_filepath=args.output_filepath,
        dna_seq_overlap=args.dna_seq_overlap,
    )
