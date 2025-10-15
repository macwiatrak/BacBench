# pip install -U transformers calflops torch
# (Optional) If you get 401s from the Hub: `huggingface-cli logout` in your shell.

import argparse

import torch
import torch.nn.functional as F
from calflops import calculate_flops
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ID = "togethercomputer/evo-1-8k-base"
REVISION = "1.1_fix"  # per model card "News" section


def _patch_hyena_fir_callable(model):
    """
    Some environments hit a signature mismatch where the engine expects a callable FIR.
    If the Hyena filter uses the fallback F.conv1d path, wrap it into a one-arg callable
    that handles permutation and groups correctly.
    """
    # Evo puts blocks under model.backbone.blocks (or model.model.blocks in some versions)
    backbone = getattr(model, "backbone", None) or getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "blocks"):
        return

    for blk in backbone.blocks:
        filt = getattr(blk, "filter", None)
        if filt is None:
            continue
        # Only patch when the FIR is the raw functional conv1d (not the custom kernel)
        if getattr(filt, "fir_fn", None) is F.conv1d and hasattr(filt, "short_filter_weight"):
            weight = filt.short_filter_weight  # [C_out, 1, K], where C_out = 3*hidden
            bias = getattr(filt, "short_filter_bias", None)  # [C_out] or None
            K = int(getattr(filt, "short_filter_length", weight.shape[-1]))

            def fir_callable(u):
                # u: [B, L, C] expected. We convert to [B, C, L], depthwise conv, then back.
                assert u.dim() == 3, f"Expected [B, L, C], got {u.shape}"
                B, L, C = u.shape
                if weight.shape[0] != C:  # noqa
                    # If shapes don't match, it's not the fallback path we intend to patch.
                    raise RuntimeError(f"Hyena FIR wrapper: channels {C} != weight.out_channels {weight.shape[0]}")  # noqa
                x = u.permute(0, 2, 1)  # [B, C, L]
                z = F.conv1d(x, weight, bias=None, stride=1, padding=K - 1, dilation=1, groups=C)  # noqa
                z = z[..., :L]  # crop to original length
                if bias is not None:  # noqa
                    z = z + bias[None, :, None]  # noqa
                return z.permute(0, 2, 1)  # back to [B, L, C]

            filt.fir_fn = fir_callable  # engine.parallel_fir will now do fir_fn(u)


def load_evo_model():
    """Load Evo model from Hugging Face with necessary patches."""
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=cfg,
        trust_remote_code=True,
        revision=REVISION,
        torch_dtype=torch.bfloat16,  # dtype doesn’t affect FLOP count
        low_cpu_mem_usage=True,
    ).eval()  # keep on CPU; we only count FLOPs
    _patch_hyena_fir_callable(model)
    return model, cfg


def flops_evo_forward(seq_len: int) -> tuple[int, int, int]:
    """Returns (forward_flops, forward_macs, params) for Evo at batch=1, seq_len=seq_len."""
    model, cfg = load_evo_model()
    vocab = int(getattr(cfg, "vocab_size", 512))

    # Dummy inputs (no tokenizer needed)
    B = 1
    inputs = {
        "input_ids": torch.randint(0, vocab, (B, seq_len), dtype=torch.long),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # calflops executes a forward pass and counts ops; keep it quiet and fast
    with torch.no_grad():
        fwd_flops, fwd_macs, params = calculate_flops(
            model=model,
            kwargs={k: v.to(device) for k, v in inputs.items()},
            include_backPropagation=False,  # forward only
            print_results=False,
            print_detailed=False,
            output_as_string=False,
        )
        return fwd_flops, fwd_macs, params


def humanize(n: float) -> str:
    """Convert a large number into a human-readable format with units."""
    for unit in ["", "K", "M", "G", "T", "P", "E"]:
        if abs(n) < 1000.0:
            return f"{n:,.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} Z"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8192, help="sequence length (context size)")
    args = ap.parse_args()
    flops, macs, params = flops_evo_forward(args.L)
    print(f"Params: {params:,}")
    print(f"Forward MACs:  {humanize(macs)} MACs")
    print(f"Forward FLOPs: {humanize(flops)} FLOPs")
