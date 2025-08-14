import itertools
from typing import Literal

import numpy as np
import pandas as pd
import torch

from bacbench.modeling.embedder import SeqEmbedder
from bacbench.modeling.utils_glm2 import preprocess_seq_for_glm2


def get_dna_seq(
    dna_seq: str,
    start: int,
    end: int,
    strand: float | None,
    promoter_len: int = 128,
) -> str:
    """Get the DNA sequence from a genome.

    Args:
        dna_seq (str): The DNA sequence to extract from.
        start (int): The start position of the sequence.
        end (int): The end position of the sequence.
        strand (float | None): The strand of the sequence. If None, the sequence is not reversed.
        promoter_len (int): The length of the promoter region to include.

    Returns
    -------
        str: The extracted DNA sequence.
    """
    seq_len = len(dna_seq)

    if strand is None:  # promoter on both sides
        seq_start = max(1, start - promoter_len)
        seq_end = min(seq_len, end + promoter_len)
    elif strand > 0:  # + strand
        seq_start = max(1, start - promoter_len)
        seq_end = min(seq_len, end)
    else:  # - strand
        seq_start = max(1, start)
        seq_end = min(seq_len, end + promoter_len)

    # Python slice: 0-based, end exclusive
    subseq = dna_seq[seq_start - 1 : seq_end]

    return subseq.upper()


def chunk_dna_sequence(dna_seq: str, max_seq_len: int, overlap: int) -> list[str]:
    """
    Chunks a DNA sequence into overlapping pieces of length `max_seq_len`.

    Args:
        dna_seq (str): The DNA sequence to chunk.
        max_seq_len (int): Maximum length of each chunk.
        overlap (int): Number of bases to overlap between consecutive chunks.

    Returns
    -------
        list[str]: List of chunked DNA substrings.
    """
    chunks = []
    n = len(dna_seq)
    start = 0

    # Keep creating chunks until we reach the end of the sequence
    while start < n:
        end = min(start + max_seq_len, n)
        chunk = dna_seq[start:end]
        chunks.append(chunk)

        # If we've already reached the end, break
        if end == n:
            break

        # Move start by (max_seq_len - overlap) to maintain overlap
        start += max_seq_len - overlap

    return chunks


def chunk_whole_genome_dna_seq(dna_sequence: str, max_seq_len: int, overlap: int) -> list[str]:
    """Chunk a whole genome DNA sequence into overlapping pieces.

    Args:
        dna_sequence (str): The whole genome DNA sequence to chunk.
        max_seq_len (int): Maximum length of each chunk.
        overlap (int): Number of bases to overlap between consecutive chunks.

    Returns
    -------
        str: The chunked DNA sequence with whitespace separating different regions.
    """
    # whitespace separates different regions, like different plasmids
    dna_sequence = dna_sequence.split()
    # chunk each region into max_seq_len chunks with overlap from both sides
    dna_sequence = [chunk_dna_sequence(item, max_seq_len=max_seq_len, overlap=overlap) for item in dna_sequence]
    # flatten list of lists
    dna_sequence = list(itertools.chain(*dna_sequence))
    return dna_sequence


def chunk_genes_dna_seqs(
    dna: str,
    start: list[int],
    end: list[int],
    strand: list[float] | None = None,
    max_seq_len: int = 512,
    promoter_len: int = 128,
    dna_seq_overlap: int = 32,
    min_seq_len: int = 32,
):
    """Chunk genes DNA sequences from a genome.

    Args:
        dna (str): The whole genome DNA sequence.
        start (List[int]): List of start positions for each gene.
        end (List[int]): List of end positions for each gene.
        strand (List[float] | None): List of strands for each gene. If None, the sequence is not reversed.
        max_seq_len (int): Maximum length of each sequence.
        promoter_len (int): Length of the promoter region to include.
        dna_seq_overlap (int): The overlap between chunks of the DNA sequence.
        min_seq_len (int): Minimum length of the sequence to keep after chunking.

    Returns
    -------
        list[str]: List of chunked DNA sequences for each gene.
    """
    if strand is None:
        strand = [None] * len(start)

    gene_df = pd.DataFrame(
        {
            "dna_sequence": [
                get_dna_seq(dna_seq=dna, start=int(s), end=int(e), strand=strand_val, promoter_len=promoter_len)
                for s, e, strand_val in zip(start, end, strand, strict=True)
            ]
        }
    )
    gene_df["dna_seq_len"] = gene_df["dna_sequence"].apply(len)
    gene_df["gene_idx"] = range(len(gene_df))
    # chunk each gene into max_seq_len chunks
    gene_df["dna_sequence"] = gene_df["dna_sequence"].apply(
        lambda x: chunk_dna_sequence(x, max_seq_len=max_seq_len, overlap=dna_seq_overlap)
    )
    # explode the list of chunks into separate rows
    gene_df = gene_df.explode("dna_sequence")
    gene_df["dna_seq_len"] = gene_df["dna_sequence"].apply(len)
    # filter out chunks that are too short
    gene_df = gene_df[gene_df["dna_seq_len"] >= min_seq_len]
    # sort by sequence length for speed up
    gene_df = gene_df.sort_values(by="dna_seq_len", ascending=False).reset_index(drop=True)
    gene_indices = gene_df["gene_idx"].tolist()
    dna_chunks = gene_df["dna_sequence"].tolist()
    return dna_chunks, gene_indices


def generate_dna_embeddings(
    embedder: SeqEmbedder,
    dna_sequence: list[str],
    batch_size: int = 128,
    max_seq_len: int = 2048,
) -> list[np.ndarray]:
    """Generate DNA embeddings using pretrained models.

    Args:
        embedder: SeqEmbedder object to embed the sequence data.
        dna_sequence (str): The DNA sequence to embed.
        batch_size (int): The batch size to use for embedding.
        max_seq_len (int): The maximum sequence length for the model.

    Returns
    -------
        list[np.ndarray]: The generated DNA embeddings.
    """
    # Initialize an empty list to store the protein embeddings
    mean_dna_embeddings = []

    # Process the DNA sequences in batches
    for i in range(0, len(dna_sequence), batch_size):
        batch_sequences = dna_sequence[i : i + batch_size]
        with torch.no_grad():
            dna_representations = embedder(batch_sequences, max_seq_len, pooling="mean")

        # Append the generated embeddings to the list
        mean_dna_embeddings += dna_representations

    return mean_dna_embeddings


def embed_genome_dna_sequences(
    embedder: SeqEmbedder,
    dna: str,
    start: list[int] | None = None,
    end: list[int] | None = None,
    strand: list[float] | None = None,
    batch_size: int = 64,
    max_seq_len: int = 1024,
    dna_seq_overlap: int = 32,
    promoter_len: int = 128,
    genome_pooling_method: Literal["mean", "max"] = None,
) -> list[np.ndarray] | np.ndarray:
    """Embed genome DNA sequences using pretrained models.

    Args:
        embedder: SeqEmbedder object to embed the sequence data.
        dna (str): The DNA sequence to embed.
        start (List[int] | None): List of start positions for each gene. If None, the whole genome is embedded.
        end (List[int] | None): List of end positions for each gene. If None, the whole genome is embedded.
        strand (List[float] | None): List of strands for each gene. If None, the sequence is not reversed.
        batch_size (int): The batch size to use for embedding.
        max_seq_len (int): The maximum sequence length for the model.
        dna_seq_overlap (int): The overlap between chunks of the DNA sequence.
        promoter_len (int): The length of the promoter region to include.
        genome_pooling_method (str): The pooling method to use for the genome level embedding.
            If None, list of DNA embedding chunks is returned.
    """
    assert isinstance(dna, str), "DNA sequence overlap must be a string"
    assert len(dna) > 0, "DNA sequence cannot be empty"

    # if start and end are not provided, we assume we want to embed the whole genome
    if start is None or end is None:
        if embedder.model_type == "glm2":
            # GLM2 model requires the whole genome to be embedded in a specific manner
            dna = preprocess_seq_for_glm2(dna_sequence=dna, max_seq_len=max_seq_len, n_overlap=dna_seq_overlap)
        else:
            dna = chunk_whole_genome_dna_seq(dna_sequence=dna, max_seq_len=max_seq_len, overlap=dna_seq_overlap)
        gene_indices = None
    # embed the dna sequence for each gene
    else:
        dna, gene_indices = chunk_genes_dna_seqs(
            dna=dna,
            start=start,
            end=end,
            strand=strand,
            promoter_len=promoter_len,
            max_seq_len=max_seq_len,
            dna_seq_overlap=dna_seq_overlap,
        )

    # embed protein sequences
    dna_embeddings = generate_dna_embeddings(
        embedder=embedder,
        dna_sequence=dna,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    # if we pool all the embeddings at genome level, we don't care about the order and we just
    # pool and return avg embeddings
    if genome_pooling_method is not None:
        dna_embeddings = np.stack(dna_embeddings)
        if genome_pooling_method == "mean":
            return np.mean(dna_embeddings, axis=0)
        if genome_pooling_method == "max":
            return np.max(dna_embeddings, axis=0)
        raise ValueError(f"Unsupported genome pooling method: {genome_pooling_method}")
    else:
        # if we don't pool, we return the list of embeddings and the gene indices
        gene_df = pd.DataFrame({"gene_idx": gene_indices, "dna_embedding": dna_embeddings})
        gene_df = gene_df.groupby("gene_idx")["dna_embedding"].apply(list)
        gene_df["dna_embeddings"] = gene_df.apply(lambda x: np.mean(x, axis=0))
        dna_embeddings = gene_df["dna_embeddings"].tolist()

    return dna_embeddings
