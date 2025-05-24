import logging
import multiprocessing as mp  # NEW
import os  # NEW: multiprocessing helpers
from typing import Any, Literal

import pandas as pd
import torch
from datasets import Dataset, IterableDataset, load_dataset
from tap import Tap
from tqdm import tqdm
from transformers import AutoModel

from bacbench.modeling.embed_prot_seqs import (
    compute_genome_protein_embeddings,
    load_plm,
)
from bacbench.modeling.utils import get_prot_seq_col_name

############################################
# ──────────────────────────────────────────
#   HELPER FUNCTIONS (unchanged + new)
# ──────────────────────────────────────────
############################################


def _iterable_to_dataframe(iter_ds: IterableDataset, max_rows: int | None = None) -> pd.DataFrame:
    """Consume an IterableDataset and materialise it as a pandas DataFrame."""
    rows: list[dict] = []
    for idx, row in enumerate(tqdm(iter_ds)):
        if max_rows is not None and idx >= max_rows:
            break
        rows.append(row)
    return pd.DataFrame.from_records(rows)


# NEW: in‑worker globals will be filled by _init_worker so model objects are
#      *shared* by forked child processes (zero‑copy on Linux/WSL/macOS)
_MODEL = None
_TOKENIZER = None
_BACFORMER = None


def _init_worker(model, tokenizer, bacformer):  # NEW
    """Save heavy objects in each process without re‑loading from disk."""
    global _MODEL, _TOKENIZER, _BACFORMER
    _MODEL = model
    _TOKENIZER = tokenizer
    _BACFORMER = bacformer


def _embed_batch(
    batch: dict[str, list[Any]],
    prot_seq_col: str,
    output_col: str,
    model_type: Literal["esm2", "esmc", "protbert"],
    batch_size: int,
    max_prot_seq_len: int,
    genome_pooling_method: str | None,
):  # NEW
    """CPU‑side tokenisation, GPU‑side forward in mixed precision."""
    with torch.cuda.amp.autocast():
        emb = compute_genome_protein_embeddings(
            model=_MODEL,
            tokenizer=_TOKENIZER,
            protein_sequences=batch[prot_seq_col],
            contig_ids=batch.get("contig_name", None),
            model_type=model_type,
            batch_size=batch_size,
            max_prot_seq_len=max_prot_seq_len,
            genome_pooling_method=genome_pooling_method,
        )
    return {output_col: emb}


def _materialise_shard(args):  # NEW helper so Pool.map can pickle
    shard_ds, prot_seq_col, output_col, model_type, batch_size, max_prot_seq_len, genome_pooling_method = args

    # batched map *inside* the shard → no python per‑row overhead
    shard_ds = shard_ds.map(
        lambda batch: _embed_batch(
            batch,
            prot_seq_col=prot_seq_col,
            output_col=output_col,
            model_type=model_type,
            batch_size=batch_size,
            max_prot_seq_len=max_prot_seq_len,
            genome_pooling_method=genome_pooling_method,
        ),
        batched=True,
        batch_size=batch_size,
        remove_columns=[prot_seq_col],
    )

    return _iterable_to_dataframe(shard_ds)


############################################
# ──────────────────────────────────────────
#   MAIN RUN LOGIC (patched for IterableDataset)
# ──────────────────────────────────────────
############################################


def run(
    dataset: Dataset | None,
    model_path: str,
    model_type: Literal["esm2", "esmc", "protbert", "bacformer"],
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    device: str | None = None,
    output_col: str = "embeddings",
    genome_pooling_method: Literal["mean", "max"] | None = None,
    max_n_proteins: int = 6000,
    max_n_contigs: int = 1000,
    streaming: bool = False,
    n_workers: int = os.cpu_count() or 1,  # NEW: user‑tunable parallellism
):
    """Embed protein sequences – optimised for IterableDataset."""
    # CHANGED: allow callers to pass a str *or* already‑loaded dataset
    if isinstance(dataset, str):
        dataset = load_dataset(dataset, streaming=streaming, cache_dir=None)

    # device logic
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bacformer handling (unchanged)
    bacformer_model = None
    if model_type == "bacformer":
        logging.info("Bacformer model used – loading Bacformer + ESM-2 base model …")
        bacformer_model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(torch.bfloat16).to(device)
        )
        model_type = "esm2"
        model_path = "facebook/esm2_t12_35M_UR50D"

    # load pLM and keep on GPU *before* forking so workers inherit the weights
    model, tokenizer = load_plm(model_path, model_type)

    ############################################
    # MAIN LOOP OVER DATA SPLITS
    ############################################
    dfs: list[pd.DataFrame] = []

    for split_name, split_ds in dataset.items():
        prot_col = get_prot_seq_col_name(split_ds.column_names)

        if isinstance(split_ds, IterableDataset):
            # ────────────────────────────────────────────────
            #   STREAMING DATASET –  PARALLEL SHARDING
            # ────────────────────────────────────────────────
            logging.info(f"Processing iterable split '{split_name}' with {n_workers} worker(s)…")

            ctx = mp.get_context("fork" if os.name != "nt" else "spawn")  # fork speeds up weight sharing
            with ctx.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(model, tokenizer, bacformer_model),
            ) as pool:
                shards = [split_ds.shard(num_shards=n_workers, index=i) for i in range(n_workers)]
                args_list = [
                    (
                        shard,
                        prot_col,
                        output_col,
                        model_type,
                        batch_size,
                        max_prot_seq_len,
                        genome_pooling_method if bacformer_model is None else None,
                    )
                    for shard in shards
                ]
                # parallel embedding; each worker returns a DataFrame
                dfs_parts = pool.map(_materialise_shard, args_list)
            df = pd.concat(dfs_parts, ignore_index=True)

        else:
            # ────────────────────────────────────────────────
            #   REGULAR IN‑MEMORY DATASET (kept for completeness)
            # ────────────────────────────────────────────────
            logging.info(f"Processing standard split '{split_name}' with num_proc={n_workers} …")

            split_ds = split_ds.map(
                lambda batch: _embed_batch(
                    batch,
                    prot_seq_col=prot_col,  # noqa
                    output_col=output_col,
                    model_type=model_type,
                    batch_size=batch_size,
                    max_prot_seq_len=max_prot_seq_len,
                    genome_pooling_method=genome_pooling_method if bacformer_model is None else None,
                ),
                batched=True,
                batch_size=batch_size,
                num_proc=n_workers,  # CHANGED: real parallelism for tokeniser
                remove_columns=[prot_col],
            )
            df = split_ds.to_pandas()

        # add split identifier
        df["split"] = split_name
        dfs.append(df)

    # merge splits, drop technical index
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.drop(columns=[c for c in ["__index_level_0__"] if c in full_df.columns], inplace=True)
    return full_df


############################################
# ──────────────────────────────────────────
#   CLI WRAPPER (same interface + new flag)
# ──────────────────────────────────────────
############################################


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    dataset_name: str
    streaming: bool = False
    output_filepath: str
    model_path: str
    model_type: Literal["esm2", "esmc", "protbert", "bacformer"]
    batch_size: int = 64
    max_prot_seq_len: int = 1024
    device: str | None = None
    output_col: str = "embeddings"
    genome_pooling_method: Literal["mean", "max"] | None = None
    max_n_proteins: int = 9000
    max_n_contigs: int = 1000
    n_workers: int = os.cpu_count() or 1  # NEW CLI argument


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    ds = load_dataset(args.dataset_name, streaming=args.streaming, cache_dir=None)

    df = run(
        dataset=ds,
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
        n_workers=args.n_workers,
    )

    df.to_parquet(args.output_filepath)
