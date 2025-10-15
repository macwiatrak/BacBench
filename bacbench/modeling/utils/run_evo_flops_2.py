# pip install -U transformers deepspeed torch

import argparse

import torch
import torch.nn.functional as F
from deepspeed.profiling.flops_profiler import FlopsProfiler
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ID = "togethercomputer/evo-1-8k-base"
REVISION = "1.1_fix"  # model card recommends this revision


def _patch_hyena_fir_callable(model):
    """Make the short-FIR path callable(u) to avoid conv1d signature pitfalls."""
    backbone = getattr(model, "backbone", None) or getattr(model, "model", None)
    if not backbone or not hasattr(backbone, "blocks"):
        return
    for blk in backbone.blocks:
        filt = getattr(blk, "filter", None)
        if filt is None:
            continue
        if getattr(filt, "fir_fn", None) is F.conv1d and hasattr(filt, "short_filter_weight"):
            w = filt.short_filter_weight  # [C, 1, K] where C=3*hidden
            b = getattr(filt, "short_filter_bias", None)
            K = int(getattr(filt, "short_filter_length", w.shape[-1]))

            def fir_callable(u):
                # u: [B, L, C] -> depthwise conv -> [B, L, C]
                B, L, C = u.shape
                x = u.permute(0, 2, 1)  # [B, C, L]
                z = F.conv1d(x, w, bias=None, stride=1, padding=K - 1, groups=C)  # noqa
                z = z[..., :L]
                if b is not None:  # noqa
                    z = z + b[None, :, None]  # noqa
                return z.permute(0, 2, 1)

            filt.fir_fn = fir_callable


def load_evo(revision=REVISION, device="cuda", dtype=torch.bfloat16):
    """Load Evo model from Hugging Face with necessary patches."""
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, revision=revision)
    # keep caches and extras off to reduce working set
    cfg.inference_mode = True
    cfg.use_cache = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, config=cfg, trust_remote_code=True, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        .eval()
        .to(device)
    )
    _patch_hyena_fir_callable(model)
    return model, cfg


def forward_flops_deepspeed(L: int) -> tuple[int, int, int]:
    """Calculate forward pass FLOPs, MACs, and params for Evo model at sequence length L."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_evo(device=device)
    vocab = int(getattr(cfg, "vocab_size", 512))
    inputs = {
        "input_ids": torch.randint(0, vocab, (1, L), device=device, dtype=torch.long),
        # Passing these avoids extra work/memory:
        "use_cache": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
    }

    prof = FlopsProfiler(model)
    torch.cuda.empty_cache()
    with torch.inference_mode():  # lower overhead & memory vs no_grad
        prof.start_profile()
        _ = model(**inputs)
        prof.stop_profile()

    fwd_flops = prof.get_total_flops()
    fwd_macs = prof.get_total_macs()
    params = prof.get_total_params()
    prof.end_profile()
    return fwd_flops, fwd_macs, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=8192, help="sequence length (context size)")
    args = parser.parse_args()
    flops, macs, params = forward_flops_deepspeed(args.L)
    print(f"Params: {params:,}")
    print(f"Forward MACs:  {macs / 1e12:.3f} TMACs")
    print(f"Forward FLOPs: {flops / 1e12:.3f} TFLOPs")
