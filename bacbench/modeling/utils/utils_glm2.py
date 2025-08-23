from __future__ import annotations

from collections import defaultdict

import numpy as np
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


def preprocess_whole_genome_for_glm2(
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


# ----------------------------- #
# 1) Precompute elements + map  #
# ----------------------------- #


def precompute_glm2_elements(
    prot_seqs: list[str],
    dna_seq: str,
    start: list[int],  # 1-based inclusive
    end: list[int],  # 1-based inclusive
    strand: list[float],
) -> tuple[list[dict], dict[int, int]]:
    """
    Build a linearized element list mixing DNA (lowercase) and PROT (UPPERCASE),
    each prefixed by a strand token, and a mapping: original gene_idx -> element_idx.

    Returns
    -------
    elements : List[dict]
        Each element has:
          - type: "DNA" or "PROT"
          - seq:  string (DNA lowercase / AA uppercase)
          - stoken: "<+>" or "<->" (DNA always "<+>")
          - gene_idx: original index for PROT, else None
    gene_idx_to_elem_idx : Dict[int, int]
        Maps original gene index → index into `elements` where that gene's PROT lives.
    """
    assert len(prot_seqs) == len(start) == len(end)
    # small hotfix for strand mismatch in 1 genome
    if len(strand) != len(prot_seqs):
        strand = list(strand) + [-1.0] * (len(prot_seqs) - len(strand))
    L = len(dna_seq)

    def stoken_for_strand(s: float) -> str:
        return "<+>" if s >= 0 else "<->"

    # Keep original indices, sort by genomic start
    genes = sorted(
        [(i, int(start[i]), int(end[i]), strand[i], (prot_seqs[i] or "")) for i in range(len(prot_seqs))],
        key=lambda t: t[1],
    )

    elements: list[dict] = []
    gene_idx_to_elem_idx: dict[int, int] = {}

    # Left flank DNA (before first gene)
    first_start = genes[0][1]
    if first_start > 1:
        left = dna_seq[: first_start - 1].lower()
        if left:
            elements.append({"type": "DNA", "seq": left, "stoken": "<+>", "gene_idx": None})

    # Interleave PROT and intergenic DNA
    for k, (orig_i, _, en_i, sgn, aa) in enumerate(genes):
        # PROT element
        prot_el = {
            "type": "PROT",
            "seq": aa.upper(),
            "stoken": stoken_for_strand(sgn),
            "gene_idx": orig_i,
        }
        elements.append(prot_el)
        gene_idx_to_elem_idx[orig_i] = len(elements) - 1

        # DNA between this and next gene
        if k < len(genes) - 1:
            next_start = genes[k + 1][1]
            # 0-based slices: gene occupies [st_i-1 : en_i); intergenic is [en_i : next_start-1)
            s = max(en_i, 0)
            e = max(next_start - 1, 0)
            if e > s:
                igs = dna_seq[s:e].lower()
                if igs:
                    elements.append({"type": "DNA", "seq": igs, "stoken": "<+>", "gene_idx": None})

    # Right flank DNA (after last gene)
    last_end = genes[-1][2]
    if last_end < L:
        right = dna_seq[last_end:].lower()
        if right:
            elements.append({"type": "DNA", "seq": right, "stoken": "<+>", "gene_idx": None})

    return elements, gene_idx_to_elem_idx


# ---------------------------------------------- #
# 2) Render one gene with balanced context, on demand
# ---------------------------------------------- #


def preprocess_glm2_gene_seq(
    elements: list[dict],
    gene_idx_to_elem_idx: dict[int, int],
    gene_idx: int,
    max_seq_len: int = 4096,
) -> tuple[str, np.array]:
    """
    Center `gene_idx`'s protein element, add left/right context until `max_seq_len`
    tokens are reached. Returns gLM2-formatted string and token-level gene mask.

    Token accounting:
      - Each element contributes 1 token for its stoken ("<+>" or "<->") + 1 token per character of `seq`.
      - When truncating an element, we always keep the stoken, and take the nearest letters:
          * LEFT side: take sequence suffix (closest to the gene)
          * RIGHT side: take sequence prefix (closest to the gene)

    Mask semantics:
      - 1 for the focal gene's amino-acid tokens only (not its stoken),
      - 0 for all DNA tokens, all other PROT tokens, and all stokens/pad.

    If padding is needed, a DNA pad element is appended: "<+>" + "n" * K (all masked 0).
    """

    # --- helpers ---
    def take_elem(side: str, eidx: int, tokens_left: int):
        """
        Take as many tokens as possible from element `eidx` on `side` ('left'|'right').
        Returns (part_string, part_mask, tokens_used) or None if no space.
        """
        if tokens_left <= 0:
            return None
        el = elements[eidx]
        st = el["stoken"]
        seq = el["seq"]

        # full element fits
        full_len = 1 + len(seq)
        if full_len <= tokens_left:
            # mask: stoken=0, seq=0 (context)
            return st + seq, [0] + [0] * len(seq), full_len

        # partial: keep stoken, then nearest letters
        k = max(tokens_left - 1, 0)  # number of sequence chars that fit
        if k == 0:
            return st, [0], 1

        if side == "left":
            letters = seq[-k:]
        else:  # right
            letters = seq[:k]
        return st + letters, [0] + [0] * len(letters), 1 + len(letters)

    # Center element index
    if gene_idx not in gene_idx_to_elem_idx:
        raise KeyError(f"gene_idx {gene_idx} not found in mapping.")
    center_eidx = gene_idx_to_elem_idx[gene_idx]
    center_el = elements[center_eidx]
    if center_el["type"] != "PROT":
        raise ValueError("Center element for gene must be PROT.")

    # Center piece (focal gene)
    center_str = center_el["stoken"] + center_el["seq"]
    center_mask = [0] + [1] * len(center_el["seq"])  # only AA tokens are 1

    total_tokens = len(center_mask)
    left_parts, left_masks = [], []
    right_parts, right_masks = [], []
    left_tokens = 0
    right_tokens = 0

    L = len(elements)
    l = center_eidx - 1
    r = center_eidx + 1

    # Grow context until we hit max_seq_len
    while total_tokens < max_seq_len and (l >= 0 or r < L):
        prefer_left = left_tokens <= right_tokens
        did = False
        # try preferred side first, then the other
        for side_name, idx in (
            (("left", l) if prefer_left else ("right", r)),
            (("right", r) if prefer_left else ("left", l)),
        ):
            if side_name == "left" and idx >= 0:
                avail = max_seq_len - total_tokens
                part = take_elem("left", idx, avail)
                if part is not None:
                    s_part, s_mask, used = part
                    left_parts.append(s_part)
                    left_masks.append(s_mask)
                    left_tokens += used
                    total_tokens += used
                    l -= 1
                    did = True
                    break
            elif side_name == "right" and idx < L:
                avail = max_seq_len - total_tokens
                part = take_elem("right", idx, avail)
                if part is not None:
                    s_part, s_mask, used = part
                    right_parts.append(s_part)
                    right_masks.append(s_mask)
                    right_tokens += used
                    total_tokens += used
                    r += 1
                    did = True
                    break
        if not did:
            break

    # Pad if still short: DNA pad "<+>" + "n" * K
    if total_tokens < max_seq_len:
        pad_tokens = max_seq_len - total_tokens
        if pad_tokens == 1:
            right_parts.append("<+>")
            right_masks.append([0])
        else:
            right_parts.append("<+>" + ("n" * (pad_tokens - 1)))
            right_masks.append([0] + [0] * (pad_tokens - 1))

    # Assemble: left side collected outward->inward, so reverse to restore genomic order
    left_str = "".join(left_parts[::-1])
    left_mask = sum(left_masks[::-1], [])  # flatten lists
    right_str = "".join(right_parts)
    right_mask = sum(right_masks, [])

    seq_str = left_str + center_str + right_str
    gene_mask = left_mask + center_mask + right_mask

    if len(gene_mask) > max_seq_len:
        # If we exceed max_seq_len, truncate the right side
        seq_str = seq_str[: max_seq_len + 2]  # +2 for the stoken
        gene_mask = gene_mask[:max_seq_len]

    special_token_counts = seq_str.count("<+>") + seq_str.count("<->")
    assert len(gene_mask) == len(seq_str) - special_token_counts * 2

    return seq_str, np.array(gene_mask)
