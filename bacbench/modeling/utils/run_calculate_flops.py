from collections import Counter
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import torch
from calflops import calculate_flops
from tap import Tap
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel

from bacbench.modeling.embed_dna import chunk_whole_genome_dna_seq
from bacbench.modeling.embedder import SeqEmbedder, load_seq_embedder

# from bacbench.modeling.utils.run_evo_flops_2 import forward_flops_deepspeed_evo
from bacbench.modeling.utils.utils_glm2 import preprocess_whole_genome_for_glm2

try:
    from bacformer.modeling.modeling_updated import BacformerCGForMaskedGM
except ImportError:
    pass


@dataclass
class TransformerShape:
    """Dataclass for Transformer model shape parameters."""

    layers: int
    d_model: int
    d_ff: int
    n_heads: int
    vocab: int  # may be 0 for models like Bacformer aggregators


def _swiglu_ffn_dim_from_cfg(cfg) -> int | None:
    # gLM2 uses SwiGLU with hidden size ~= (2/3)*4d, rounded up to multiple_of
    # https://huggingface.co/tattabio/gLM2_650M (config/impl shows swiglu_multiple_of & ffn_dim_multiplier)
    multiple_of = getattr(cfg, "swiglu_multiple_of", None)
    d = getattr(cfg, "dim", None)
    mult = getattr(cfg, "ffn_dim_multiplier", None)
    if d is None or multiple_of is None:
        return None
    hidden = int((8 * d) / 3)  # (2/3)*4d
    if mult is not None:
        hidden = int(hidden * mult)
    # round up to multiple_of
    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
    return hidden


def extract_shape_from_config(cfg) -> TransformerShape:
    """Extract Transformer shape parameters from a Hugging Face config object."""
    # Standard names (BERT/ESM/most): num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, vocab_size
    if hasattr(cfg, "num_hidden_layers"):
        L = int(cfg.num_hidden_layers)
        D = int(cfg.hidden_size)
        F = int(getattr(cfg, "intermediate_size", 4 * D))
        H = int(cfg.num_attention_heads)
        V = int(getattr(cfg, "vocab_size", 0) or 0)
        return TransformerShape(L, D, F, H, V)

    # GLM2-style configs ("dim","depth","heads", SwiGLU FFN dims)
    if hasattr(cfg, "depth") and hasattr(cfg, "dim") and hasattr(cfg, "heads"):
        L = int(cfg.depth)
        D = int(cfg.dim)
        H = int(cfg.heads)
        F = _swiglu_ffn_dim_from_cfg(cfg) or (4 * D)  # fallback
        V = int(getattr(cfg, "vocab_size", 0) or 0)
        return TransformerShape(L, D, F, H, V)

    # Fallbacks (try common aliases)
    L = int(getattr(cfg, "n_layer", 0) or getattr(cfg, "num_layers", 0))
    D = int(getattr(cfg, "d_model", 0) or getattr(cfg, "hidden_size", 0))
    F = int(getattr(cfg, "intermediate_size", 4 * D) or (4 * D))
    H = int(getattr(cfg, "n_head", 0) or getattr(cfg, "num_attention_heads", 0))
    V = int(getattr(cfg, "vocab_size", 0) or 0)
    return TransformerShape(L, D, F, H, V)


def model_states_bytes(
    num_params: int,
    *,
    weight_bytes: int = 2,  # bf16/fp16 weights
    grad_bytes: int = 2,  # bf16/bf16 grads (set 4 if you keep fp32 grads)
    master_weight_bytes: int = 4,  # fp32 master weights
    adam_state_bytes: int = 8,  # Adam m & v in fp32 (≈8 bytes/param)
) -> dict[str, int]:
    """Calculate model state memory in bytes for training."""
    # DeepSpeed docs: Adam optimizer states use ~8 bytes per parameter (4B momentum + 4B variance).
    # Weights in bf16 are 2B, master fp32 4B; grads commonly bf16 (2B) or fp32 (4B). :contentReference[oaicite:0]{index=0}
    return {
        "weights": num_params * weight_bytes,
        "grads": num_params * grad_bytes,
        "master_weights": num_params * master_weight_bytes,
        "optimizer_states": num_params * adam_state_bytes,
    }


# ---------- Training activation memory with FlashAttention (bytes) ----------
def activations_bytes_flash(
    T: int,
    shape: TransformerShape,
    *,
    bytes_per_elem: int = 2,  # bf16/float16 activations
    checkpoint_ratio: float = 1.0,  # 1.0 = save acts for all layers; <1.0 = activation checkpointing
    save_residual: bool = True,
    save_qkv: bool = True,  # set False for Hyena-style layers (Evo); True for standard attention
    save_mlp_act: bool = True,
) -> int:
    """
    With FlashAttention we do *not* materialize the T×T attention matrix; saved activations scale ~O(T·D) and O(T·F),
    not O(T^2). Peak training memory ≈ (#saved layers) × (per-layer saved activations). :contentReference[oaicite:1]{index=1}
    """
    L, D, F = shape.layers, shape.d_model, shape.d_ff
    per_layer_tokens = 0
    if save_residual:
        per_layer_tokens += T * D
    if save_qkv:
        per_layer_tokens += 3 * T * D
    if save_mlp_act:
        per_layer_tokens += T * F
    saved_layers = max(1, int(round(L * checkpoint_ratio)))
    return per_layer_tokens * saved_layers * bytes_per_elem


def activations_bytes_naive_segments(
    lengths: list[int],
    shape: TransformerShape,
    *,
    bytes_per_elem: int = 2,
    checkpoint_ratio: float = 1.0,
    save_residual: bool = True,
    save_qkv: bool = True,
    save_mlp_act: bool = True,
) -> int:
    """
    Naïve attention: if attention is ONLY within proteins, the quadratic term is sum_i (H * L_i^2).
    This corrects the earlier (sum L_i)^2 assumption.
    Saved activations per layer ≈ [residual/QKV/MLP (linear in sum L_i)] + [attention probs/scores (∑ L_i^2)].
    """
    L, D, F, H = shape.layers, shape.d_model, shape.d_ff, shape.n_heads
    T = int(sum(lengths))
    sum_sq = int(sum(li * li for li in lengths))
    per_layer_elems = 0
    if save_residual:
        per_layer_elems += T * D
    if save_qkv:
        per_layer_elems += 3 * T * D
    if save_mlp_act:
        per_layer_elems += T * F
    # attention matrices per head (scores/probs) — ONLY if you’re actually saving them (non-Flash)
    per_layer_elems += H * sum_sq
    saved_layers = max(1, int(round(L * checkpoint_ratio)))
    return per_layer_elems * saved_layers * bytes_per_elem


def human_bytes(n: int) -> str:
    """Convert bytes to a human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:,.2f} {u}"
        x /= 1024
    return f"{x:.2f} ZB"


def calculate_genome_flops(
    input_seqs: list[str],  # protein or DNA sequences
    embedder: SeqEmbedder,
    max_seq_len: int,
    bp_factor: float = 2.0,  # backprop factor (2.0 for Adam, 1.0 for SGD)
    model_name_or_path: str | None = None,  # used to fetch config when embedder.model is None
    checkpoint_ratio: float = 1.0,
):
    """Calculate flops for a model on a list of protein or DNA sequences."""
    # Treat the genome as one long example: T = sum(lengths)
    # total_tokens = sum(len(s) for s in input_seqs)

    # to speed up calculation, do a counter of sequence lens
    input_seq_counts = [len(seq) for seq in input_seqs]
    input_seqs_counter = Counter([len(seq) for seq in input_seqs])
    # for each len in the counter, get a representative sequence from input_seqs
    input_seqs = {len(seq): seq for seq in input_seqs}
    n_params_seen = None

    output = []
    for seq_len, seq in tqdm(input_seqs.items()):
        if embedder.model_type == "evo":
            # fwd_flops, fwd_macs, params = forward_flops_deepspeed_evo(len(seq))
            pass
        else:
            seqs = embedder._preprocess_seqs([seq])
            inputs = embedder._tokenize(seqs, max_seq_len=max_seq_len)
            if embedder.model_type in ["glm2"]:
                inputs = {"input_ids": inputs["input_ids"].to(embedder.device)}
            with torch.no_grad():
                fwd_flops, fwd_macs, params = calculate_flops(
                    model=embedder.model,
                    kwargs=inputs,  # your inputs dict
                    include_backPropagation=False,  # forward only
                    print_results=False,
                    print_detailed=False,
                    output_as_string=False,
                )
        n_params_seen = n_params_seen or int(params)
        train_flops = fwd_flops * (1.0 + bp_factor)
        train_macs = fwd_macs * (1.0 + bp_factor)
        output.append(
            {
                "n_training_params": int(params),
                "fwd_MACs": fwd_macs * input_seqs_counter[seq_len],
                "fwd_FLOPs": fwd_flops * input_seqs_counter[seq_len],
                "fwd+bwd_MACs": train_macs * input_seqs_counter[seq_len],
                "fwd+bwd_FLOPs": train_flops * input_seqs_counter[seq_len],
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
    # -------- Peak training memory for ONE full-genome batch (B=1, T=sum(lengths)) --------
    # cfg is assumed present (either via embedder.model.config or via model_name_or_path)
    if getattr(embedder, "model", None) is not None and hasattr(embedder.model, "config"):
        cfg = embedder.model.config
    else:
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    shape = extract_shape_from_config(cfg)

    P = overall_stats["n_params"]
    states = model_states_bytes(
        P, weight_bytes=2, grad_bytes=2, master_weight_bytes=4, adam_state_bytes=8
    )  # bf16 everywhere; Adam states in fp32 (8 bytes/param).

    # FlashAttention: no O(T^2) saved activations. For Evo (Hyena) we can skip QKV saves.
    save_qkv = embedder.model_type != "evo"
    # acts = activations_bytes_flash(
    #     total_tokens,
    #     shape,
    #     bytes_per_elem=2,
    #     checkpoint_ratio=checkpoint_ratio,
    #     save_residual=True,
    #     save_qkv=save_qkv,
    #     save_mlp_act=True,
    # )  # FA reduces saved-activation memory to ~O(T·D), not O(T^2).

    acts = activations_bytes_naive_segments(
        input_seq_counts,
        shape,
        bytes_per_elem=2,
        checkpoint_ratio=checkpoint_ratio,
        save_residual=True,
        save_qkv=save_qkv,
        save_mlp_act=True,
    )  # FA reduces saved-activation memory to ~O(T·D), not O(T^2).

    peak_train_bytes = int(acts + sum(states.values()))
    overall_stats.update(
        {
            "peak_train_mem_bytes": peak_train_bytes,
            "peak_train_mem_human": human_bytes(peak_train_bytes),
            "activation_mem_bytes": int(acts),
            "activation_mem_human": human_bytes(int(acts)),
            "state_mem_bytes": int(sum(states.values())),
            "state_mem_human": human_bytes(int(sum(states.values()))),
        }
    )
    return overall_stats


def calculate_genome_bacformer(
    input_seqs: list[str],  # protein or DNA sequences
    model: nn.Module,
    max_n_proteins: int = 9000,
    bp_factor: float = 2.0,  # backprop factor (2.0 for Adam, 1.0 for SGD)
    bacformer_model_type: Literal["base", "large"] = "base",
    checkpoint_ratio: float = 1.0,
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
    # -------- Peak training memory: B=1, T = N proteins; Bacformer has no vocab head --------
    cfg = model.config  # you said cfg is never None
    shape = extract_shape_from_config(cfg)
    # For Bacformer aggregators, ignore vocab in memory (no LM head). (shape.vocab may be 0 anyway.)

    P = int(params)
    states = model_states_bytes(
        P, weight_bytes=2, grad_bytes=2, master_weight_bytes=4, adam_state_bytes=8
    )  # bf16 + Adam states in fp32.

    acts = activations_bytes_flash(
        n_proteins,
        shape,
        bytes_per_elem=2,
        checkpoint_ratio=checkpoint_ratio,
        save_residual=True,
        save_qkv=True,
        save_mlp_act=True,
    )  # FA activation model (no O(T^2) storage). :contentReference[oaicite:6]{index=6}

    peak_train_bytes = int(acts + sum(states.values()))

    return {
        "n_params": int(params),
        "total_flops_fwd": int(fwd_flops),
        "total_flops_fwd+bwd": int(train_flops),
        "total_macs_fwd": int(fwd_macs),
        "total_macs_fwd+bwd": int(train_macs),
        "peak_train_mem_bytes": peak_train_bytes,
        "peak_train_mem_human": human_bytes(peak_train_bytes),
        "activation_mem_bytes": int(acts),
        "activation_mem_human": human_bytes(int(acts)),
        "state_mem_bytes": int(sum(states.values())),
        "state_mem_human": human_bytes(int(sum(states.values()))),
    }


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
    # load the data
    df = pd.read_parquet(input_df_path)
    out = []
    for _, row in df[4:].iterrows():
        if model_type in ["protein", "bacformer"]:
            input_seqs = row["protein_sequences"]
        elif model_type == "dna":
            input_seqs = " ".join(row["dna_sequence"])
            input_seqs = chunk_whole_genome_dna_seq(
                dna_sequence=input_seqs, max_seq_len=max_seq_len, overlap=dna_seq_overlap
            )
        else:
            # process the data into gLM2 format
            input_seqs = " ".join(row["dna_sequence"])
            input_seqs = preprocess_whole_genome_for_glm2(
                dna_sequence=input_seqs, max_seq_len=max_seq_len, n_overlap=dna_seq_overlap
            )

        if model_type != "bacformer":
            embedder = load_seq_embedder(model_name_or_path)
            if embedder.model_type == "evo":
                embedder.model = None  # free up memory
            stats = calculate_genome_flops(
                input_seqs=input_seqs,
                embedder=embedder,
                max_seq_len=max_seq_len,
                model_name_or_path=model_name_or_path,
                checkpoint_ratio=1.0,
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if "bacformer-300m" in model_name_or_path.lower():
                bacformer_model = (
                    BacformerCGForMaskedGM.from_pretrained(model_name_or_path)
                    .bacformer.eval()
                    .to(torch.bfloat16)
                    .to(device)
                )
                prot_model_path = "Synthyra/ESMplusplus_small"
                bacformer_model_type = "large"
            else:
                bacformer_model = (
                    AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
                    .eval()
                    .to(torch.bfloat16)
                    .to(device)
                )
                prot_model_path = "facebook/esm2_t12_35M_UR50D"
                bacformer_model_type = "base"
            embedder = load_seq_embedder(prot_model_path)
            plm_stats = calculate_genome_flops(
                input_seqs=input_seqs,
                embedder=embedder,
                max_seq_len=max_seq_len,
                model_name_or_path=model_name_or_path,
                checkpoint_ratio=1.0,
            )
            stats = calculate_genome_bacformer(
                input_seqs=input_seqs,
                model=bacformer_model,
                max_n_proteins=max_n_proteins,
                bacformer_model_type=bacformer_model_type,
                checkpoint_ratio=1.0,
            )
            stats["total_flops_fwd+bwd"] = stats["total_flops_fwd"] + plm_stats["total_flops_fwd+bwd"]
            stats["total_flops_fwd"] += plm_stats["total_flops_fwd"]
            stats["total_macs_fwd+bwd"] = stats["total_macs_fwd"] + plm_stats["total_macs_fwd+bwd"]
            stats["total_macs_fwd"] += plm_stats["total_macs_fwd"]
            stats["n_params"] = plm_stats["n_params"]
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
