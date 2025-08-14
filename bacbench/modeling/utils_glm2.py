from __future__ import annotations

from collections import defaultdict

import pandas as pd

from bacbench.pp import dna_seq_to_cds_and_intergenic


# -------------------------
# Helpers: sequence cleanup
# -------------------------
def _sanitize_cds(seq: str) -> str:
    """Uppercase protein; map unexpected chars to 'X'."""
    allowed = set("ACDEFGHIKLMNPQRSTVWYXBZUO")  # standard + common extras
    s = (seq or "").upper()
    return "".join(ch if ch in allowed else "X" for ch in s)


def _sanitize_igs(seq: str, keep_unk: bool = False) -> str:
    """Lowercase DNA; optionally map non-atcg to 'a' to avoid unknowns."""
    s = (seq or "").lower()
    if keep_unk:
        return s
    return "".join(ch if ch in {"a", "t", "c", "g"} else "a" for ch in s)


# ---------------------------------------------
# DF -> per-contig gLM2 strings (no tokenizers)
# ---------------------------------------------
def df_to_glm2_contig_texts(df: pd.DataFrame, igs_use_plus_token: bool = True) -> dict[int, str]:
    """
    Build raw gLM2 strings per contig, respecting element order.
    Required columns: sequence, start, end, strand, sequence_type, contig_idx
    Assumes df is sorted by ["contig_idx","start"].
    """
    required = {"sequence", "start", "end", "strand", "sequence_type", "contig_idx"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")
    # ensure order
    df = df.sort_values(["contig_idx", "start"], kind="mergesort").reset_index(drop=True)

    by_contig: dict[int, list[str]] = defaultdict(list)
    for row in df.itertuples(index=False):
        seq_type = row.sequence_type
        strand = int(row.strand)
        contig_idx = int(row.contig_idx)
        seq = row.sequence

        if seq_type == "cds":
            elem = _sanitize_cds(seq)
            prefix = "<+>" if strand >= 0 else "<->"
        elif seq_type == "intergenic":
            # gLM2 example uses <+> for IGS; keep that convention
            elem = _sanitize_igs(seq, keep_unk=False)
            prefix = "<+>" if igs_use_plus_token else ("<+>" if strand >= 0 else "<->")
        else:
            raise ValueError(f"Unknown sequence_type: {seq_type}")

        by_contig[contig_idx].append(prefix + elem)

    return {cid: "".join(parts) for cid, parts in by_contig.items()}


def get_last_strand_token(s: str) -> str | None:
    """Return the last occurrence of '<+' or '<-' in s, or None if neither is present."""
    i_plus = s.rfind("<+")
    i_minus = s.rfind("<-")
    if i_plus == -1 and i_minus == -1:
        return None
    if i_plus > i_minus:
        return "<+>"
    return "<->"


def clean_glm2_chunk(chunk: str) -> str:
    """
    Clean a gLM2 chunk by removing leading/trailing whitespace and ensuring
    it does not start with a strand token.
    """
    if chunk.startswith("+") or chunk.startswith("-"):
        return chunk[2:]
    if chunk.startswith(">"):
        return chunk[1:]
    return chunk


def chunk_glm2_seqs(
    contig_texts: dict[int, str],
    max_chars: int = 4096,
    n_overlap: int = 64,
) -> list[str]:
    """
    Chunk gLM2 strings into fixed-size character windows *per contig*.

    - Overlap: consecutive windows overlap by `n_overlap` characters (except
      possibly the final window, which may overlap more due to end-alignment).
    - No alignment to element boundaries (pure character slicing).
    - Final window is end-aligned: if the last slice would be shorter than
      `max_chars`, shift left so the window ends at the contig end for more context.

    Notes
    -----
      * Requires a helper `get_last_strand_token(chunk: str) -> str | None`
        that returns '<+' or '<-' (or None). If present, it's prepended to
        the next window to carry strand context across cuts.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if n_overlap < 0:
        raise ValueError("n_overlap must be >= 0")
    if n_overlap >= max_chars:
        raise ValueError("n_overlap must be < max_chars")

    windows: list[str] = []

    for _, text in sorted(contig_texts.items()):
        if not text:
            continue

        last_strand_token = ""
        step = max_chars - n_overlap
        n = len(text)
        start = 0

        while start < n:
            # Is this the final chunk for this contig?
            is_last = (start + max_chars) >= n

            # End-align the final window if it would be short.
            if is_last and (n - start) < max_chars:
                start = max(0, n - max_chars)

            chunk_raw = text[start : start + max_chars]
            chunk = (last_strand_token or "") + chunk_raw
            if chunk:
                windows.append(chunk)

            # Carry the last seen strand marker into the next chunk
            tok = get_last_strand_token(chunk)
            last_strand_token = tok if tok is not None else ""

            if is_last:
                break  # emitted the final chunk for this contig

            start += step

    return windows


# ---------------------------------------
# One-stop: DF -> list[str] (gLM2 format)
# ---------------------------------------
def prepare_glm2_strings_from_df(
    df: pd.DataFrame,
    max_chars: int = 4096,
    n_overlap: int = 64,
) -> list[str]:
    """
    Convert your feature DataFrame into gLM2-ready strings and return
    overlapping character windows per contig. No tokenizer is used.
    """
    # enforce per-contig isolation by building per-contig strings first
    contig_texts = df_to_glm2_contig_texts(df)
    return chunk_glm2_seqs(contig_texts, max_chars=max_chars, n_overlap=n_overlap)


def preprocess_seq_for_glm2(
    dna_sequence: list[str] | str,
    contig_names: list[str] = None,
    max_seq_len: int = 4096,
    n_overlap: int = 64,
) -> list[str]:
    """Function to preprocess DNA sequences for gLM2.

    Args:
        dna_sequence (list[str] | str): DNA sequences to preprocess.
        contig_names (list[str], optional): Names of the contigs. Defaults to None.
        max_seq_len (int, optional): Maximum length of the sequence. Defaults to 4096.
        n_overlap (int, optional): Number of overlapping characters. Defaults to 64.

    Returns
    -------
        List[str]: List of preprocessed gLM2 strings.
    """
    df = dna_seq_to_cds_and_intergenic(dna_sequence, contig_names=contig_names)
    glm2_strings = prepare_glm2_strings_from_df(df=df, max_chars=max_seq_len, n_overlap=n_overlap)
    return glm2_strings
