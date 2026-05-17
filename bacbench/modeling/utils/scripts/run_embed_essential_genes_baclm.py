import pandas as pd
import torch
from datasets import load_dataset
from tap import Tap
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_OPERON_MAX_INTERGENIC_DISTANCE = 20


def _reverse_complement(sequence: str) -> str:
    """Return the reverse-complemented DNA sequence."""
    return sequence.translate(_DNA_COMPLEMENT)[::-1]


def _normalize_strand(strand: int | str) -> int:
    """Normalise strand annotations to +/-1 integers."""
    if isinstance(strand, str):
        strand = strand.strip()
    if strand in {"+", "1"}:
        return 1
    if strand in {"-", "-1"}:
        return -1
    return int(strand)


def _annotate_operon_promoter_sources(
    gene_df: pd.DataFrame,
    genome_col: str,
    contig_col: str,
    max_intergenic_distance: int = _OPERON_MAX_INTERGENIC_DISTANCE,
) -> pd.DataFrame:
    """Map each gene to the promoter-defining gene for its operon within a contig."""
    group_cols = [genome_col, contig_col]
    ordered_gene_df = gene_df.sort_values([genome_col, contig_col, "start", "end", "gene_idx"]).copy()
    ordered_gene_df["strand_norm"] = ordered_gene_df["strand"].apply(_normalize_strand)
    ordered_gene_df["prev_gene_end"] = ordered_gene_df.groupby(group_cols)["end"].shift(1)
    ordered_gene_df["next_gene_start"] = ordered_gene_df.groupby(group_cols)["start"].shift(-1)
    ordered_gene_df["prev_strand_norm"] = ordered_gene_df.groupby(group_cols)["strand_norm"].shift(1)
    ordered_gene_df["intergenic_distance"] = ordered_gene_df["start"] - ordered_gene_df["prev_gene_end"] - 1
    ordered_gene_df["same_operon_as_prev"] = (
        ordered_gene_df["prev_gene_end"].notna()
        & (ordered_gene_df["strand_norm"] == ordered_gene_df["prev_strand_norm"])
        & (ordered_gene_df["intergenic_distance"] < max_intergenic_distance)
    )
    ordered_gene_df["operon_idx"] = ordered_gene_df.groupby(group_cols)["same_operon_as_prev"].transform(
        lambda same_operon: (~same_operon).cumsum()
    )

    operon_df = (
        ordered_gene_df.groupby(group_cols + ["operon_idx"], sort=False)
        .agg(
            operon_strand=("strand_norm", "first"),
            first_gene_idx=("gene_idx", "first"),
            last_gene_idx=("gene_idx", "last"),
        )
        .reset_index()
    )
    operon_df["promoter_source_gene_idx"] = operon_df["first_gene_idx"].where(
        operon_df["operon_strand"] >= 0,
        operon_df["last_gene_idx"],
    )

    promoter_source_df = ordered_gene_df[["gene_idx", "start", "end", "prev_gene_end", "next_gene_start"]].rename(
        columns={
            "gene_idx": "promoter_source_gene_idx",
            "start": "promoter_source_start",
            "end": "promoter_source_end",
            "prev_gene_end": "promoter_source_prev_gene_end",
            "next_gene_start": "promoter_source_next_gene_start",
        }
    )
    operon_df = operon_df.merge(promoter_source_df, on="promoter_source_gene_idx", how="left")

    return ordered_gene_df.merge(
        operon_df[
            group_cols
            + [
                "operon_idx",
                "promoter_source_start",
                "promoter_source_end",
                "promoter_source_prev_gene_end",
                "promoter_source_next_gene_start",
            ]
        ],
        on=group_cols + ["operon_idx"],
        how="left",
    )[
        [
            "gene_idx",
            "strand_norm",
            "promoter_source_start",
            "promoter_source_end",
            "promoter_source_prev_gene_end",
            "promoter_source_next_gene_start",
        ]
    ]


def _extract_gene_flanks(
    dna_sequence: str,
    start: int,
    end: int,
    strand: int,
    flank_len: int,
    prev_gene_end: int | None,
    next_gene_start: int | None,
) -> str:
    """Return a promoter sequence bounded by neighboring CDS."""
    seq_len = len(dna_sequence)
    prev_gene_end = 0 if pd.isna(prev_gene_end) else int(prev_gene_end)
    next_gene_start = (seq_len + 1) if pd.isna(next_gene_start) else int(next_gene_start)

    if strand >= 0:
        promoter_start = max(0, start - 1 - flank_len, prev_gene_end)
        promoter_end = max(0, start - 1)
        promoter_seq = dna_sequence[promoter_start:promoter_end]
    else:
        promoter_start = min(seq_len, end)
        promoter_end = min(seq_len, end + flank_len, next_gene_start - 1)
        promoter_seq = _reverse_complement(dna_sequence[promoter_start:promoter_end])
    return promoter_seq


def _masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Pool token embeddings while ignoring padding positions."""
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def run(
    model_name: str,
    dna_dataset_path: str,
    prot_dataset_path: str,
    output_file_path: str,
    batch_size: int = 32,
    max_seq_len: int = 2048,
    promoter_len: int = 128,
    min_promoter_len: int = 3,
    label_col: str = "essential",
):
    """Run the embedding script for essential genes using BacLM."""
    # load the DNA sequences
    if dna_dataset_path.endswith(".parquet"):
        dna_df = pd.read_parquet(dna_dataset_path)
    else:
        dna_df = load_dataset(dna_dataset_path, split="test").to_pandas()

    contig_col = "contig_id" if "contig_id" in dna_df.columns else "contig_name"
    genome_col = "genome_name" if "genome_name" in dna_df.columns else "strain_name"
    # limit to necessary columns
    dna_df = dna_df[[genome_col, contig_col, "dna_sequence"]]

    # explode the col
    dna_df = dna_df.explode([contig_col, "dna_sequence"])
    # set index to genome_col and contig_col for faster dna seq lookup
    dna_df.set_index([genome_col, contig_col], inplace=True)

    if prot_dataset_path.endswith(".parquet"):
        prot_df = pd.read_parquet(
            prot_dataset_path,
            columns=[genome_col, contig_col, "start", "end", "strand", label_col, "protein_sequence"],
        )
    else:
        prot_df = load_dataset(prot_dataset_path, split="test").to_pandas()
        prot_df = prot_df[[genome_col, contig_col, "start", "end", "strand", "protein_sequence"]]
    # explode the cols
    prot_df = prot_df.explode([contig_col, "start", "end", "strand", label_col, "protein_sequence"]).explode(
        ["start", "end", "strand", label_col, "protein_sequence"]
    )
    prot_df["start"] = prot_df["start"].astype(int)
    prot_df["end"] = prot_df["end"].astype(int)
    prot_df["gene_idx"] = list(range(len(prot_df)))  # add a gene index column for easier tracking
    promoter_source_df = _annotate_operon_promoter_sources(prot_df, genome_col=genome_col, contig_col=contig_col)
    prot_df = prot_df.merge(promoter_source_df, on="gene_idx", how="left")

    seqs = []
    for _, row in prot_df.iterrows():
        genome_name = row[genome_col]
        contig_id = row[contig_col]
        # Input coordinates are 1-based inclusive. Convert to Python's 0-based,
        # end-exclusive slices only when extracting the promoter sequence.
        strand = int(row["strand_norm"])
        prot_seq = row["protein_sequence"]
        gene_idx = row["gene_idx"]
        promoter_source_start = int(row["promoter_source_start"])
        promoter_source_end = int(row["promoter_source_end"])
        promoter_source_prev_gene_end = row["promoter_source_prev_gene_end"]
        promoter_source_next_gene_start = row["promoter_source_next_gene_start"]

        # append the protein sequence
        seqs.append((prot_seq.upper(), gene_idx, "cds", len(prot_seq)))

        dna_sequence = dna_df.loc[(genome_name, contig_id), "dna_sequence"]
        promoter_seq = _extract_gene_flanks(
            dna_sequence=dna_sequence,
            start=promoter_source_start,
            end=promoter_source_end,
            strand=strand,
            flank_len=promoter_len,
            prev_gene_end=promoter_source_prev_gene_end,
            next_gene_start=promoter_source_next_gene_start,
        )

        if len(promoter_seq) >= min_promoter_len:
            seqs.append((promoter_seq.lower(), gene_idx, "promoter", len(promoter_seq)))

    # make it a pandas dataframe and sort by sequence length to minimize padding during batching
    seqs_df = pd.DataFrame(seqs, columns=["sequence", "gene_idx", "seq_type", "seq_len"])
    seqs_df = seqs_df.sort_values("seq_len", ascending=False).reset_index(drop=True)
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

    # embed the sequence using the BacLM embedder and save the results to a parquet file
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(runtime_device)
    model.eval()

    embeddings = []
    with torch.inference_mode():
        for idx in tqdm(range(0, len(seqs_df), batch_size), desc="Embedding sequences"):
            seqs = seqs_df["sequence"].iloc[idx : idx + batch_size].tolist()
            batch = tokenizer.batch_encode_plus(
                seqs,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
            batch = {k: v.to(runtime_device, non_blocking=True) for k, v in batch.items()}

            encoder_outputs = model(
                input_ids=batch["input_ids"],
                token_type_ids=batch.get("token_type_ids"),
                attention_mask=batch.get("attention_mask"),
                output_hidden_states=False,
            )

            pooled = _masked_mean_pool(encoder_outputs.last_hidden_state, batch.get("attention_mask"))
            embeddings.extend(pooled.cpu().float().tolist())

    if len(embeddings) != len(seqs_df):
        raise RuntimeError(f"Embedding count mismatch: got {len(embeddings)} embeddings for {len(seqs_df)} rows")

    seqs_df["embedding"] = embeddings

    hidden_dim = len(embeddings[0])
    zero_embedding = torch.zeros(hidden_dim, dtype=torch.float32)
    aggregated_embeddings = []
    for gene_idx, gene_seqs_df in seqs_df.groupby("gene_idx", sort=False):
        embeddings_by_type = dict(zip(gene_seqs_df["seq_type"], gene_seqs_df["embedding"], strict=False))
        cds_embedding_raw = embeddings_by_type.get("cds")
        if cds_embedding_raw is None:
            raise RuntimeError(f"Missing protein embedding for gene_idx={gene_idx}")

        cds_embedding = torch.tensor(cds_embedding_raw, dtype=torch.float32)
        promoter_embedding = (
            torch.tensor(embeddings_by_type["promoter"], dtype=torch.float32)
            if embeddings_by_type.get("promoter") is not None
            else None
        )

        available_embeddings = [cds_embedding]
        if promoter_embedding is not None:
            available_embeddings.append(promoter_embedding)

        promoter_slot = promoter_embedding if promoter_embedding is not None else zero_embedding

        aggregated_embeddings.append(
            {
                "gene_idx": gene_idx,
                "mean_embedding": torch.stack(available_embeddings, dim=0).mean(dim=0).numpy(),
                "cds_mean_embedding": cds_embedding.numpy(),
                "promoter_embedding": promoter_slot.numpy(),
                "promoter_cds_embedding": torch.cat([promoter_slot, cds_embedding], dim=0).numpy(),
            }
        )

    seqs_df = pd.DataFrame(aggregated_embeddings)
    assert len(seqs_df) == len(prot_df), (
        f"Gene index mismatch: got {len(seqs_df)} unique gene indices but expected {len(prot_df)}"
    )
    prot_df = prot_df.merge(
        seqs_df[["gene_idx", "mean_embedding", "cds_mean_embedding", "promoter_embedding", "promoter_cds_embedding"]],
        on="gene_idx",
        how="inner",
    )
    prot_df = prot_df.drop(
        columns=[
            "gene_idx",
            "strand_norm",
            "promoter_source_start",
            "promoter_source_end",
            "promoter_source_prev_gene_end",
            "promoter_source_next_gene_start",
        ]
    )

    # groupby again for compatibility
    prot_df = (
        prot_df.groupby([genome_col, contig_col]).agg(list).reset_index().groupby([genome_col]).agg(list).reset_index()
    )
    prot_df.to_parquet(output_file_path, index=False)


class ArgumentParser(Tap):
    """Argument parser for finetuning linear model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    model_name: str = "macwiatrak/baclm-350m-masked"
    output_filepath: str = "/projects/public/u6fp/benchmarks/tasks/essential-genes/updated/baclm_with_promoter.parquet"
    dna_dataset_path: str = "/projects/public/u6fp/benchmarks/tasks/essential-genes/DEG_dna_dataset.parquet"
    prot_dataset_path: str = "/projects/public/u6fp/benchmarks/tasks/essential-genes/DEG_prot_dataset.parquet"
    label_col: str = "essential"
    promoter_len: int = 128
    min_promoter_len: int = 3
    max_seq_len: int = 2048
    batch_size: int = 32


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        model_name=args.model_name,
        dna_dataset_path=args.dna_dataset_path,
        prot_dataset_path=args.prot_dataset_path,
        label_col=args.label_col,
        output_file_path=args.output_filepath,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        promoter_len=args.promoter_len,
        min_promoter_len=args.min_promoter_len,
    )
