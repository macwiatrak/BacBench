import pandas as pd
import torch
from tap import Tap
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _reverse_complement(sequence: str) -> str:
    """Return the reverse-complemented DNA sequence."""
    return sequence.translate(_DNA_COMPLEMENT)[::-1]


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
    dna_input_file_path: str,
    prot_input_file_path: str,
    output_file_path: str,
    batch_size: int = 32,
    max_seq_len: int = 2048,
    promoter_len: int = 128,
    min_promoter_len: int = 3,
):
    """Run the embedding script for essential genes using BacLM."""
    # load the DNA sequences and the train/test split information
    dna_df = pd.read_parquet(dna_input_file_path, columns=["genome_name", "contig_id", "dna_sequence"])
    # explode the col
    dna_df = dna_df.explode(["contig_id", "dna_sequence"])
    # set index to genome_name and contig_id for faster dna seq lookup
    dna_df.set_index(["genome_name", "contig_id"], inplace=True)

    prot_df = pd.read_parquet(
        prot_input_file_path,
        columns=["genome_name", "contig_id", "start", "end", "strand", "essential", "protein_sequence", "split"],
    )
    # explode the cols
    prot_df = prot_df.explode(["contig_id", "start", "end", "strand", "essential", "protein_sequence"]).explode(
        ["start", "end", "strand", "essential", "protein_sequence"]
    )
    prot_df["gene_idx"] = list(range(len(prot_df)))  # add a gene index column for easier tracking

    seqs = []
    for _, row in prot_df.iterrows():
        genome_name = row["genome_name"]
        contig_id = row["contig_id"]
        # Input coordinates are 1-based inclusive. Convert to Python's 0-based,
        # end-exclusive slices only when extracting the promoter sequence.
        start = row["start"]
        end = row["end"]
        strand = row["strand"]
        prot_seq = row["protein_sequence"]
        gene_idx = row["gene_idx"]

        # append the protein sequence
        seqs.append((prot_seq.upper(), gene_idx, "protein", len(prot_seq)))

        if isinstance(strand, str):
            strand = strand.strip()
        if strand in {"+", "1"}:
            strand = 1
        elif strand in {"-", "-1"}:
            strand = -1
        else:
            strand = int(strand)

        dna_sequence = dna_df.loc[(genome_name, contig_id), "dna_sequence"]
        if strand >= 0:
            # Positive-strand promoter is directly upstream of the gene start.
            promoter_start = max(0, start - 1 - promoter_len)
            promoter_end = max(0, start - 1)
            promoter_seq = dna_sequence[promoter_start:promoter_end]
        else:
            # Negative-strand promoter is downstream in genomic coordinates; reverse
            # complement it so the sequence is oriented relative to the gene.
            promoter_start = min(len(dna_sequence), end)
            promoter_end = min(len(dna_sequence), end + promoter_len)
            promoter_seq = _reverse_complement(dna_sequence[promoter_start:promoter_end])

        if len(promoter_seq) < min_promoter_len:
            continue
        seqs.append((promoter_seq.lower(), gene_idx, "intergenic", len(promoter_seq)))

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
    zero_embedding = [0.0] * hidden_dim
    aggregated_embeddings = []
    for gene_idx, gene_seqs_df in seqs_df.groupby("gene_idx", sort=False):
        embeddings_by_type = dict(zip(gene_seqs_df["seq_type"], gene_seqs_df["embedding"], strict=False))
        protein_embedding = embeddings_by_type.get("protein")
        if protein_embedding is None:
            raise RuntimeError(f"Missing protein embedding for gene_idx={gene_idx}")

        promoter_embedding = embeddings_by_type.get("intergenic")
        available_embeddings = [protein_embedding]
        if promoter_embedding is not None:
            available_embeddings.append(promoter_embedding)

        # Keep concat_embedding width stable: [protein_embedding, promoter_embedding].
        # Genes without a long enough promoter get a zero-filled promoter slot.
        concat_embedding = torch.cat(
            [
                torch.tensor(protein_embedding, dtype=torch.float32),
                torch.tensor(
                    promoter_embedding if promoter_embedding is not None else zero_embedding, dtype=torch.float32
                ),
            ],
            dim=0,
        )
        aggregated_embeddings.append(
            {
                "gene_idx": gene_idx,
                "mean_embedding": torch.tensor(available_embeddings, dtype=torch.float32).mean(dim=0).numpy(),
                "concat_embedding": concat_embedding.numpy(),
            }
        )

    seqs_df = pd.DataFrame(aggregated_embeddings)
    assert len(seqs_df) == len(prot_df), (
        f"Gene index mismatch: got {len(seqs_df)} unique gene indices but expected {len(prot_df)}"
    )
    prot_df = prot_df.merge(seqs_df[["gene_idx", "mean_embedding", "concat_embedding"]], on="gene_idx", how="inner")
    prot_df = prot_df.drop(columns=["gene_idx"])

    # groupby again for compatibility
    prot_df = (
        prot_df.groupby(["genome_name", "contig_id", "split"])
        .agg(list)
        .reset_index()
        .groupby(["genome_name", "split"])
        .agg(list)
        .reset_index()
    )
    prot_df.to_parquet(output_file_path, index=False)


class ArgumentParser(Tap):
    """Argument parser for finetuning linear model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    model_name: str = "macwiatrak/baclm-350m-masked"
    output_filepath: str = "/projects/public/u6fp/benchmarks/tasks/essential-genes/updated/baclm_with_promoter.parquet"
    dna_input_file_path: str = "/projects/public/u6fp/benchmarks/tasks/essential-genes/DEG_dna_dataset.parquet"
    prot_input_file_path: str = "/projects/public/u6fp/benchmarks/tasks/essential-genes/DEG_prot_dataset.parquet"
    promoter_len: int = 128
    min_promoter_len: int = 3
    max_seq_len: int = 2048
    batch_size: int = 32


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        model_name=args.model_name,
        dna_input_file_path=args.dna_input_file_path,
        prot_input_file_path=args.prot_input_file_path,
        output_file_path=args.output_filepath,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        promoter_len=args.promoter_len,
        min_promoter_len=args.min_promoter_len,
    )
