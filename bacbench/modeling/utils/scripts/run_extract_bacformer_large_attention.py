"""Extract attention weights from Bacformer Large model.

For each genome in the dataset, this script:
  1. Embeds all protein sequences with the ESM-C small pLM (Synthyra/ESMplusplus_small).
  2. Runs a forward pass through Bacformer Large with return_attn_weights=True.
  3. Saves two numpy arrays per genome into an output directory keyed by strain_name:
       - avg_attn.npy  : mean attention over all 30 layers, shape (n_proteins, n_proteins)
       - last_attn.npy : attention from the last layer only,  shape (n_heads, n_proteins, n_proteins)

Each saved matrix is restricted to the real (non-padding) protein tokens, so the
dimensions vary per genome.

Usage
-----
python bacbench/modeling/run_extract_bacformer_large_attention.py \
    --dataset-name macwiatrak/bacbench-ppi-stringdb-protein-sequences-small \
    --split validation \
    --output-dir /path/to/attention_weights \
    --model-path macwiatrak/bacformer-large-masked-complete-genomes \
    --batch-size 64 \
    --max-n-proteins 12000
"""

from __future__ import annotations

import gc
import logging
import os

import numpy as np
import torch
from bacbench.modeling.embed_prot_seqs import compute_genome_protein_embeddings
from bacbench.modeling.embedder import load_seq_embedder
from bacbench.modeling.utils.utils import get_prot_seq_col_name, protein_embeddings_to_inputs
from datasets import load_dataset
from tap import Tap
from transformers import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------


def _cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Core per-genome extraction
# ---------------------------------------------------------------------------


def extract_attention_weights(
    bacformer_model: torch.nn.Module,
    protein_embeddings: list,
    contig_ids: list,
    max_n_proteins: int = 12000,
    max_n_contigs: int = 1000,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Run one genome through Bacformer Large and return attention weights.

    Parameters
    ----------
    bacformer_model:
        Loaded BacformerLargeModel in eval mode.
    protein_embeddings:
        List[List[np.ndarray]] — per-contig lists of per-protein embeddings.
    contig_ids:
        Flat list of integer contig IDs (one per protein token after exploding).
    max_n_proteins:
        Maximum number of proteins to feed into the model.
    max_n_contigs:
        Maximum number of contigs to feed into the model.
    device:
        Target device string.

    Returns
    -------
    avg_attn : np.ndarray, shape (n_proteins, n_proteins)
        Mean attention weight matrix averaged over all layers and all heads.
    last_attn : np.ndarray, shape (n_proteins, n_proteins)
        Attention weight matrix from the final layer, averaged over all heads.
    """
    inputs = protein_embeddings_to_inputs(
        protein_embeddings=protein_embeddings,
        max_n_proteins=max_n_proteins,
        max_n_contigs=max_n_contigs,
        contig_ids=contig_ids,
        bacformer_model_type="large",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Number of real (non-padding) protein tokens
    n_real = int(inputs["attention_mask"].sum().item())

    with torch.no_grad():
        outputs = bacformer_model(
            protein_embeddings=inputs["protein_embeddings"].to(bacformer_model.dtype),
            attention_mask=inputs["attention_mask"],
            contig_ids=inputs["contig_ids"],
            return_attn_weights=True,
            return_dict=True,
        )

    # outputs.attentions is a list of length n_layers.
    # Each element has shape (1, n_heads, seq_len, seq_len) where seq_len may
    # include padding positions.  We strip padding by slicing [:n_real, :n_real].
    attentions = outputs.attentions  # list[Tensor]

    # Average over heads per layer immediately on CPU to keep memory low.
    # Each element becomes shape (n_real, n_real) after head-averaging.
    layer_tensors = []
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        # shape: (1, H, S, S) -> mean over H -> (n_real, n_real)
        layer_cpu = layer_attn[0, :, :n_real, :n_real].float().cpu().mean(dim=0)
        layer_tensors.append(layer_cpu)

    # Free GPU memory as soon as possible
    del outputs, inputs, attentions
    _cleanup_gpu()

    all_layers = torch.stack(layer_tensors, dim=0)  # (n_layers, n_real, n_real)

    # Average over all layers -> (n_real, n_real)
    avg_attn = all_layers.mean(dim=0).numpy().astype(np.float32)

    # Last layer only, heads already averaged -> (n_real, n_real)
    last_attn = all_layers[-1].numpy().astype(np.float32)

    del all_layers, layer_tensors
    gc.collect()

    return avg_attn, last_attn


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------


def run(
    dataset_name: str,
    split: str,
    output_dir: str,
    model_path: str = "macwiatrak/bacformer-large-masked-complete-genomes",
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    max_n_proteins: int = 12000,
    max_n_contigs: int = 1000,
    device: str | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
    cache_dir: str | None = None,
) -> None:
    """Extract and save attention weights for every genome in *split*.

    Outputs
    -------
    For each genome with strain_name ``<strain>`` the following files are written::

        <output_dir>/<split>/<strain>/avg_attn.npy      # (n_proteins, n_proteins)
        <output_dir>/<split>/<strain>/last_attn.npy     # (n_heads,    n_proteins, n_proteins)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Bacformer Large
    # ------------------------------------------------------------------
    logger.info("Loading Bacformer Large from %s …", model_path)
    bacformer_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(torch.bfloat16).to(device)

    # ------------------------------------------------------------------
    # 2. Load pLM embedder (ESM-C small, same as used during pretraining)
    # ------------------------------------------------------------------
    plm_path = "Synthyra/ESMplusplus_small"
    logger.info("Loading pLM embedder from %s …", plm_path)
    embedder = load_seq_embedder(plm_path)

    # ------------------------------------------------------------------
    # 3. Load dataset split
    # ------------------------------------------------------------------
    logger.info("Loading dataset %s split=%s …", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    if start_idx is not None or end_idx is not None:
        begin = start_idx or 0
        end = end_idx if end_idx is not None else len(dataset)
        dataset = dataset.select(range(begin, min(end, len(dataset))))
    logger.info("Processing %d genomes.", len(dataset))

    prot_col = get_prot_seq_col_name(dataset.column_names)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Per-genome loop
    # ------------------------------------------------------------------
    for idx, row in enumerate(dataset):
        strain_name = str(row.get("strain_name", f"genome_{idx}"))

        logger.info("[%d/%d] %s — embedding proteins …", idx + 1, len(dataset), strain_name)

        # ---- Step A: embed protein sequences with pLM -------------------
        try:
            protein_embeddings = compute_genome_protein_embeddings(
                embedder=embedder,
                protein_sequences=row[prot_col],
                contig_ids=row.get("contig_name", None),
                batch_size=batch_size,
                max_prot_seq_len=max_prot_seq_len,
                genome_pooling_method=None,  # keep per-protein embeddings
            )
        except BaseException as exc:  # noqa: BLE001
            logger.error("[%d/%d] %s — pLM embedding failed: %s", idx + 1, len(dataset), strain_name, exc)
            _cleanup_gpu()
            continue

        _cleanup_gpu()

        # ---- Step B: build contig_ids flat list -------------------------
        # protein_embeddings is List[List[np.ndarray]] (one inner list per contig)
        if isinstance(protein_embeddings[0], np.ndarray):
            # already flat (single contig genome)
            protein_embeddings = [protein_embeddings]

        contig_ids_flat = []
        for contig_idx, contig in enumerate(protein_embeddings):
            contig_ids_flat.extend([contig_idx] * len(contig))

        # ---- Step C: run Bacformer Large with attention output ----------
        logger.info("[%d/%d] %s — running Bacformer Large …", idx + 1, len(dataset), strain_name)
        try:
            avg_attn, last_attn = extract_attention_weights(
                bacformer_model=bacformer_model,
                protein_embeddings=protein_embeddings,
                contig_ids=contig_ids_flat,
                max_n_proteins=max_n_proteins,
                max_n_contigs=max_n_contigs,
                device=device,
            )
        except BaseException as exc:  # noqa: BLE001
            logger.error(
                "[%d/%d] %s — Bacformer forward failed: %s",
                idx + 1,
                len(dataset),
                strain_name,
                exc,
            )
            _cleanup_gpu()
            continue

        # ---- Step D: save --------------------------------------------
        np.save(os.path.join(output_dir, f"{strain_name}_avg_attn.npy"), avg_attn)
        np.save(os.path.join(output_dir, f"{strain_name}_last_attn.npy"), last_attn)
        logger.info(
            "[%d/%d] %s — saved avg_attn %s, last_attn %s",
            idx + 1,
            len(dataset),
            strain_name,
            avg_attn.shape,
            last_attn.shape,
        )

        # ---- Cleanup between genomes --------------------------------
        del avg_attn, last_attn, protein_embeddings, contig_ids_flat
        _cleanup_gpu()

    logger.info("Done. Results saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class ArgumentParser(Tap):
    """Argument parser for extracting Bacformer Large attention weights."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    dataset_name: str  # HuggingFace dataset name
    split: str  # dataset split, e.g. "validation" or "test"
    output_dir: str  # directory to save per-genome attention arrays
    model_path: str = "macwiatrak/bacformer-large-masked-complete-genomes"
    batch_size: int = 64  # batch size for pLM embedding
    max_prot_seq_len: int = 1024  # max protein sequence length for pLM
    max_n_proteins: int = 12000  # max proteins per genome for Bacformer
    max_n_contigs: int = 1000  # max contigs per genome for Bacformer
    device: str | None = None  # e.g. "cuda" or "cpu"; auto-detected if None
    start_idx: int | None = None  # start index for slicing the dataset
    end_idx: int | None = None  # end index for slicing the dataset
    cache_dir: str | None = None  # HuggingFace cache directory


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=args.output_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_prot_seq_len=args.max_prot_seq_len,
        max_n_proteins=args.max_n_proteins,
        max_n_contigs=args.max_n_contigs,
        device=args.device,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        cache_dir=args.cache_dir,
    )
