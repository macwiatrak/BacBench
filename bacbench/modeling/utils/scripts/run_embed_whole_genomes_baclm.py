import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tap import Tap
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from bacbench.modeling.utils.utils import _slice_split
from bacbench.pp import dna_seq_to_cds_and_intergenic


def _masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    special_tokens_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool token embeddings while ignoring padding and special tokens."""
    if attention_mask is None:
        mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=hidden_states.dtype)
    else:
        mask = attention_mask.to(hidden_states.dtype)

    if special_tokens_mask is not None:
        mask = mask * (~special_tokens_mask.bool()).to(hidden_states.dtype)

    mask = mask.unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _get_contig_sequences(dna_sequence: list[str] | str) -> list[str]:
    """Normalise the dataset DNA field to a list of contig strings."""
    if isinstance(dna_sequence, str):
        return dna_sequence.split()
    return dna_sequence


def _aggregate_region_embeddings(embeddings: list[np.ndarray], hidden_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and max over region embeddings, guarding against empty groups."""
    if len(embeddings) == 0:
        zeros = np.zeros(hidden_dim, dtype=np.float32)
        return zeros, zeros

    stacked = np.asarray(embeddings, dtype=np.float32)
    return stacked.mean(axis=0), stacked.max(axis=0)


def run(
    dataset_name: str,
    model_name_or_path: str,
    output_dir: str,
    batch_size: int = 32,
    max_seq_len: int = 2048,
    min_seq_len: int = 3,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> pd.DataFrame:
    """Embed CDS and intergenic regions of each genome and save genome-level summaries."""
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = _slice_split(dataset, start_idx, end_idx)

    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    hidden_dim = int(model.config.hidden_size)

    output = []
    for example_idx, example in tqdm(enumerate(dataset), desc="Embedding genomes"):
        genome_name = example["genome_name"]
        contig_sequences = _get_contig_sequences(example["dna_sequence"])
        genome_df = dna_seq_to_cds_and_intergenic(contig_sequences)
        genome_df["seq_len"] = genome_df["sequence"].apply(len)
        genome_df = (
            genome_df[genome_df["seq_len"] >= min_seq_len]
            .sort_values("seq_len", ascending=False)
            .reset_index(drop=True)
        )

        if genome_df.empty:
            zero_embedding = np.zeros(hidden_dim, dtype=np.float32)
            output.append(
                {
                    "genome_name": genome_name,
                    "cds_mean_embedding": zero_embedding,
                    "cds_max_embedding": zero_embedding,
                    "intergenic_mean_embedding": zero_embedding,
                    "intergenic_max_embedding": zero_embedding,
                    "mean_embedding": zero_embedding,
                    "max_embedding": zero_embedding,
                }
            )
            continue

        genome_embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for batch_start in range(0, len(genome_df), batch_size):
                seqs = genome_df["sequence"].iloc[batch_start : batch_start + batch_size].tolist()
                batch = tokenizer.batch_encode_plus(
                    seqs,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                )
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

                encoder_outputs = model(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch.get("token_type_ids"),
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=False,
                )

                pooled = _masked_mean_pool(
                    encoder_outputs.last_hidden_state,
                    batch.get("attention_mask"),
                    batch.get("special_tokens_mask"),
                )
                genome_embeddings.extend(pooled.cpu().float().numpy())

        if len(genome_embeddings) != len(genome_df):
            raise RuntimeError(
                f"Embedding count mismatch for {genome_name}: got {len(genome_embeddings)} embeddings for {len(genome_df)} rows"
            )

        genome_df["embedding"] = genome_embeddings

        cds_mean_embedding, cds_max_embedding = _aggregate_region_embeddings(
            genome_df.loc[genome_df["sequence_type"] == "cds", "embedding"].tolist(),
            hidden_dim=hidden_dim,
        )
        intergenic_mean_embedding, intergenic_max_embedding = _aggregate_region_embeddings(
            genome_df.loc[genome_df["sequence_type"] == "intergenic", "embedding"].tolist(),
            hidden_dim=hidden_dim,
        )

        # Mean/max over region embeddings is intentional here: every CDS/intergenic region
        # contributes equally, regardless of its raw sequence length.
        all_region_embeddings = np.asarray(genome_df["embedding"].tolist(), dtype=np.float32)
        mean_embedding = all_region_embeddings.mean(axis=0)
        max_embedding = all_region_embeddings.max(axis=0)

        output.append(
            {
                "genome_name": genome_name,
                "example_idx": example_idx,
                "cds_mean_embedding": cds_mean_embedding,
                "cds_max_embedding": cds_max_embedding,
                "intergenic_mean_embedding": intergenic_mean_embedding,
                "intergenic_max_embedding": intergenic_max_embedding,
                "mean_embedding": mean_embedding,
                "max_embedding": max_embedding,
            }
        )

    output_df = pd.DataFrame(output)
    end_idx_for_name = -100 if end_idx is None else end_idx
    output_df.to_parquet(os.path.join(output_dir, f"chunk_{start_idx}_{end_idx_for_name}.parquet"), index=False)
    return output_df


class ArgumentParser(Tap):
    """Argument parser for whole-genome BacLM embedding."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    dataset_name: str
    model_name_or_path: str
    output_dir: str
    batch_size: int = 32
    max_seq_len: int = 2048
    min_seq_len: int = 3
    start_idx: int = 0
    end_idx: int | None = None


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        dataset_name=args.dataset_name,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
