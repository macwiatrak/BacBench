from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import torch

from bacbench.modeling.embedder import SeqEmbedder
from bacbench.modeling.utils.utils import protein_embeddings_to_inputs


def generate_protein_embeddings(
    embedder: SeqEmbedder,
    protein_sequences: list[str],
    batch_size: int = 64,
    max_seq_len: int = 1024,
) -> list[np.ndarray]:
    """Generate protein embeddings using pretrained models.

    Args:
        embedder (SeqEmbedder): object to embed the protein sequences with a pLM model.
        protein_sequences (List[str]): List of protein sequences to generate embeddings for.
        batch_size (int): Batch size for processing sequences.
        max_seq_len (int): Maximum sequence length for the model.
    :return: List[np.ndarray]: List of protein embeddings.
    """
    # Initialize an empty list to store the protein embeddings
    mean_protein_embeddings = []

    # Process the protein sequences in batches
    for i in range(0, len(protein_sequences), batch_size):
        batch_sequences = protein_sequences[i : i + batch_size]

        with torch.no_grad():
            protein_representations = embedder(
                batch_sequences,
                max_seq_len=max_seq_len,
                pooling="mean",
            )

        # Append the generated embeddings to the list, moving them to CPU and converting to numpy
        mean_protein_embeddings += protein_representations

    return mean_protein_embeddings


def compute_genome_protein_embeddings(
    embedder: SeqEmbedder,
    protein_sequences: list[str] | list[list[str]],
    contig_ids: list[str] = None,
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    genome_pooling_method: Literal["mean", "max"] = None,
) -> list[np.ndarray] | np.ndarray:
    """Embed genome protein sequences using pretrained models.

    Args:
        embedder (SeqEmbedder): object to embed the protein sequences with a pLM model.
        protein_sequences (List[str]): List or list of lists of protein sequences to generate embeddings for.
        batch_size (int): Batch size for processing sequences.
        max_seq_len (int): Maximum sequence length for the model.
        protein_pooling_method (str): Pooling method to use on protein level, either "mean" or "cls".
        genome_pooling_method (str): Pooling method to use on genome level, either "mean" or "cls".
    :return: List[np.ndarray]: List of protein embeddings.
    """
    assert len(protein_sequences) > 0, "Protein sequence list is empty, please include proteins in the list"

    # if the list of protein sequences is not nested, make it nested
    if isinstance(protein_sequences[0], str):
        protein_sequences = [protein_sequences]

    if contig_ids is not None:
        assert len(protein_sequences) == len(contig_ids), "Length of protein sequences and contig IDs must match"
    else:
        # create dummy contig ids to make it work in the next step
        contig_ids = [0] * len(protein_sequences)

    # create and explode dataframe
    prot_seqs_df = pd.DataFrame(
        {
            "contig_id": contig_ids,
            "protein_sequence": protein_sequences,
        }
    )
    # get contig order which will be useful later
    prot_seqs_df["contig_idx"] = range(len(prot_seqs_df))
    prot_seqs_df = prot_seqs_df.explode("protein_sequence")
    # get protein order which will be useful later
    prot_seqs_df["protein_index"] = range(len(prot_seqs_df))
    # get protein sequence length
    prot_seqs_df["prot_len"] = prot_seqs_df["protein_sequence"].apply(len)
    # sort by protein length, this is important for the model inference speedup
    prot_seqs_df = prot_seqs_df.sort_values(by="prot_len")

    # embed protein sequences
    protein_embeddings = generate_protein_embeddings(
        embedder=embedder,
        protein_sequences=prot_seqs_df["protein_sequence"].tolist(),
        batch_size=batch_size,
        max_seq_len=max_prot_seq_len,
    )

    # if we pool all the embeddings at genome level, we don't care about the order and we just
    # pool and return avg embeddings
    if genome_pooling_method is not None:
        protein_embeddings = np.stack(protein_embeddings)
        if genome_pooling_method == "mean":
            return np.mean(protein_embeddings, axis=0)
        if genome_pooling_method == "max":
            return np.max(protein_embeddings, axis=0)
        raise ValueError(f"Unsupported genome pooling method: {genome_pooling_method}")

    # if we pool at protein level, we need to return the embeddings in the same order as the input
    prot_seqs_df["protein_embedding"] = protein_embeddings
    # sort by protein index
    prot_seqs_df = prot_seqs_df.sort_values(by="protein_index")
    # group by contig id and get the list of protein embeddings
    prot_seqs_df = prot_seqs_df.groupby(["contig_id", "contig_idx"])["protein_embedding"].apply(list).reset_index()
    # sort by contig index and drop it, as it is not needed anymore
    prot_seqs_df = prot_seqs_df.sort_values(by="contig_idx").drop(columns=["contig_idx"])
    # convert to list of lists
    protein_embeddings = prot_seqs_df["protein_embedding"].tolist()
    return protein_embeddings


def compute_bacformer_embeddings(
    model: Callable,
    protein_embeddings: list[list[np.ndarray]] | list[np.ndarray],
    contig_ids: list[str] = None,
    max_n_proteins: int = 6000,
    max_n_contigs: int = 1000,
    genome_pooling_method: Literal["mean", "max"] = None,
    prot_emb_idx: int = 4,
) -> np.ndarray:
    """Compute Bacformer embeddings for a list of protein embeddings.

    Args:
        model (BacformerModel): The Bacformer model to use for embedding.
        protein_embeddings (List[List[np.ndarray]]): The protein embeddings to compute the Bacformer embeddings for.
        contig_ids: (List[str]): List of contig ids (names) the protein embeddings belong to.
        max_n_proteins (int): The maximum number of proteins to use for each genome.
        max_n_contigs (int): The maximum number of contigs to use for each genome.
        genome_pooling_method (str): The pooling method to use for the genome level embedding.
        prot_emb_idx (int): protein embedding token idx for Bacformer (default to 4).

    Returns
    -------
        List[np.ndarray]: The Bacformer embeddings for the input protein embeddings.
    """
    assert len(protein_embeddings) > 0, "Protein sequence list is empty, please include proteins in the list"

    # if the list of protein sequences is not nested, make it nested
    if isinstance(protein_embeddings[0], np.ndarray):
        protein_embeddings = [protein_embeddings]

    if contig_ids is not None:
        assert len(protein_embeddings) == len(contig_ids), "Length of protein sequences and contig IDs must match"
    else:
        # create dummy contig ids to make it work in the next step
        contig_ids = [0] * len(protein_embeddings)

    # create and explode dataframe
    prot_embs_df = pd.DataFrame(
        {
            "contig_id": contig_ids,
            "protein_embedding": protein_embeddings,
        }
    )
    # get contig order which will be useful later
    prot_embs_df["contig_idx"] = range(len(prot_embs_df))
    prot_embs_df = prot_embs_df.explode("protein_embedding")

    # get model inputs
    device = model.device
    inputs = protein_embeddings_to_inputs(
        protein_embeddings=protein_embeddings,
        max_n_proteins=max_n_proteins,
        max_n_contigs=max_n_contigs,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Compute Bacformer embeddings
    with torch.no_grad():
        bacformer_embeddings = model(
            protein_embeddings=inputs["protein_embeddings"].type(model.dtype),
            special_tokens_mask=inputs["special_tokens_mask"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        ).last_hidden_state

    # perform genome pooling
    if genome_pooling_method == "mean":
        return bacformer_embeddings.mean(dim=1).type(torch.float32).cpu().squeeze().numpy()
    elif genome_pooling_method == "max":
        return bacformer_embeddings.max(dim=1).values.type(torch.float32).cpu().squeeze().numpy()

    # only keep the protein embeddings and not special tokens
    bacformer_embeddings = bacformer_embeddings[inputs["special_tokens_mask"] == prot_emb_idx]
    # make it into a list
    bacformer_embeddings = list(bacformer_embeddings.type(torch.float32).cpu().numpy())

    # if the number of embeddings is less than the number of proteins, pad with None
    if len(bacformer_embeddings) != len(prot_embs_df):
        bacformer_embeddings += [None] * (len(prot_embs_df) - len(bacformer_embeddings))
    prot_embs_df["protein_embedding"] = bacformer_embeddings
    # group by contig id and get the list of protein embeddings
    prot_embs_df = prot_embs_df.groupby(["contig_id", "contig_idx"])["protein_embedding"].apply(list).reset_index()
    # sort by contig index and drop it, as it is not needed anymore
    prot_embs_df = prot_embs_df.sort_values(by="contig_idx").drop(columns=["contig_idx"])
    # convert to list of lists
    protein_embeddings = prot_embs_df["protein_embedding"].tolist()
    return protein_embeddings
