import numpy as np


def prepare_gene_seqs_for_evo(
    dna: str,
    start: list[int],
    end: list[int],
    strand: list[float],
    max_seq_len: int = 8192,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Prepare the gene sequences for the evo model by extracting the gene and adding
    equal left/right genomic context so each output sequence has length max_seq_len.

    Notes
    -----
    - `start`/`end` are assumed 1-based inclusive coordinates (common in genomics).
    - If the gene itself is >= max_seq_len, we truncate (head if at start of contig else tail).
    - If near contig edges, we borrow remaining context from the opposite side.
    - If still short (e.g., tiny contig), we pad with 'N' to reach max_seq_len and mark those as 0 in the mask.
    - `strand` is accepted but not used to reverse complement; sequences are in reference genome orientation.
    """
    dna_len = len(dna)
    dna_seqs: list[str] = []
    gene_masks: list[np.ndarray] = []

    for st_i, en_i, sd in zip(start, end, strand, strict=False):
        # safety clamp
        st_i = max(1, min(st_i, dna_len))
        en_i = max(1, min(en_i, dna_len))
        if en_i < st_i:
            st_i, en_i = en_i, st_i  # ensure st_i <= en_i

        # 0-based slice for the gene (Python slicing end is exclusive)
        gene_seq = dna[st_i - 1 : en_i]
        gene_len = len(gene_seq)

        # Case 1: gene alone already >= max length -> truncate
        if gene_len >= max_seq_len:
            if sd == 1:
                gene_seq = gene_seq[:max_seq_len]
            else:
                gene_seq = gene_seq[-max_seq_len:]
            dna_seqs.append(gene_seq)
            gene_masks.append(np.ones(len(gene_seq), dtype=int))
            continue

        # Desired remaining context to fill
        remaining = max_seq_len - gene_len
        left_desired = remaining // 2
        right_desired = remaining - left_desired

        # Max available on each side within contig
        max_left = st_i - 1  # number of bases to the left of start
        max_right = dna_len - en_i  # number of bases to the right of end

        # Take what we can
        left_len = min(left_desired, max_left)
        right_len = min(right_desired, max_right)

        # If we couldn't get enough on one side, borrow from the other
        deficit = remaining - (left_len + right_len)
        if deficit > 0:
            # try to extend left
            take_left = min(deficit, max_left - left_len)
            left_len += take_left
            deficit -= take_left
        if deficit > 0:
            # then extend right
            take_right = min(deficit, max_right - right_len)
            right_len += take_right
            deficit -= take_right

        # Extract contexts (0-based indices)
        left_start = (st_i - 1) - left_len
        left_end = st_i - 1
        right_start = en_i
        right_end = en_i + right_len

        left_context = dna[left_start:left_end] if left_len > 0 else ""
        right_context = dna[right_start:right_end] if right_len > 0 else ""

        # If still short (tiny contig), pad with 'N' equally
        built_len = len(left_context) + gene_len + len(right_context)
        pad_deficit = max_seq_len - built_len
        if pad_deficit > 0:
            pad_left = pad_deficit // 2
            pad_right = pad_deficit - pad_left
            left_context = ("N" * pad_left) + left_context
            right_context = right_context + ("N" * pad_right)

        gene_seq_w_context = left_context + gene_seq + right_context
        gene_mask = [0] * len(left_context) + [1] * gene_len + [0] * len(right_context)

        # Final safety
        assert len(gene_seq_w_context) == max_seq_len, (
            f"Built sequence length {len(gene_seq_w_context)} != max_seq_len {max_seq_len}"
        )
        assert len(gene_seq_w_context) == len(gene_mask), "Sequence and mask must match lengths."

        dna_seqs.append(gene_seq_w_context)
        gene_masks.append(np.array(gene_mask, dtype=int))

    return dna_seqs, gene_masks
