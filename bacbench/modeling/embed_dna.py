import itertools
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


def load_dna_lm(
    model_path: str, model_type: Literal["nucleotide_transformer", "mistral_dna", "dnabert2"]
) -> tuple[Callable, Callable]:
    """Load a pretrained DNA model.

    Args:
        model_path (str): The path to the pretrained model.
        model_type (str): The type of model to load.

    Returns
    -------
        Callable: The loaded model.
    """
    if model_type.lower() not in ["nucleotide_transformer", "mistral_dna", "dnabert2"]:
        raise ValueError(
            "Model currently not supported, please choose out of available models: ['nucleotide_transformer', 'mistral_dna']"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type.lower() == "nucleotide_transformer":
        model = (
            AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval().to(torch.float16)
        )
    elif model_type.lower() in ["mistral_dna", "dnabert2"]:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def chunk_dna_sequence(dna_seq: str, max_seq_len: int = 1024, overlap: int = 128) -> list[str]:
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


def generate_dna_embeddings(
    model: Callable,
    tokenizer: Callable,
    dna_sequence: str,
    model_type: Literal["nucleotide_transformer", "mistral_dna", "dnabert2"],
    batch_size: int = 64,
    max_seq_len: int = 1024,
    dna_seq_overlap: int = 32,
) -> list[np.ndarray]:
    """Generate DNA embeddings using pretrained models.

    Args:
        model (Callable): The pretrained model to use for embedding.
        tokenizer (Callable): The tokenizer to use for embedding.
        dna_sequence (str): The DNA sequence to embed.
        model_type (str): The type of model to use for embedding.
        batch_size (int): The batch size to use for embedding.
        max_seq_len (int): The maximum sequence length for the model.
        dna_seq_overlap (int): The overlap between chunks of the DNA sequence.

    Returns
    -------
        list[np.ndarray]: The generated DNA embeddings.
    """
    # Initialize an empty list to store the protein embeddings
    mean_dna_embeddings = []

    # get model device
    device = model.device

    # split the DNA sequence if whitespace is present
    # whitespace separates different regions, like different plasmids
    dna_sequence = dna_sequence.split()
    # chunk each region into max_seq_len chunks with overlap from both sides
    dna_sequence = [chunk_dna_sequence(item, max_seq_len=max_seq_len, overlap=dna_seq_overlap) for item in dna_sequence]
    # flatten list of lists
    dna_sequence = list(itertools.chain(*dna_sequence))
    # Process the DNA sequences in batches
    for i in range(0, len(dna_sequence), batch_size):
        batch_sequences = dna_sequence[i : i + batch_size]

        # tokenize the input
        inputs = tokenizer.batch_encode_plus(
            batch_sequences, return_tensors="pt", padding="longest", truncation=True, max_length=max_seq_len
        )
        # move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the last hidden state from the model
        with torch.no_grad():
            if model_type == "nucleotide_transformer":
                # Get the last hidden state from the model
                last_hidden_state = model(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    encoder_attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )["hidden_states"][-1]
            elif model_type == "mistral_dna":
                last_hidden_state = model(
                    input_ids=inputs["input_ids"],
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=inputs["attention_mask"],
                ).last_hidden_state
            elif model_type == "dnabert2":
                last_hidden_state = model(
                    input_ids=inputs["input_ids"],
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=inputs["attention_mask"],
                )[0]

            dna_representations = torch.einsum(
                "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
            ) / inputs["attention_mask"].sum(1).unsqueeze(1)

        # Append the generated embeddings to the list, moving them to CPU and converting to numpy
        mean_dna_embeddings += list(dna_representations.cpu().numpy())

    return mean_dna_embeddings


def embed_genome_dna_sequences(
    model: Callable,
    tokenizer: Callable,
    dna_sequence: str,
    model_type: Literal["nucleotide_transformer", "mistral_dna", "dnabert2"],
    batch_size: int = 64,
    max_seq_len: int = 1024,
    dna_seq_overlap: int = 32,
    genome_pooling_method: Literal["mean", "max"] = None,
) -> list[np.ndarray] | np.ndarray:
    """Embed genome DNA sequences using pretrained models.

    Args:
        model (Callable): The pretrained model to use for embedding.
        tokenizer (Callable): The tokenizer to use for embedding.
        dna_sequence (str): The DNA sequence to embed.
        model_type (str): The type of model to use for embedding.
        batch_size (int): The batch size to use for embedding.
        max_seq_len (int): The maximum sequence length for the model.
        dna_seq_overlap (int): The overlap between chunks of the DNA sequence.
        genome_pooling_method (str): The pooling method to use for the genome level embedding.
            If None, list of DNA embedding chunks is returned.
    """
    assert isinstance(dna_sequence, str), "DNA sequence overlap must be a string"
    assert len(dna_sequence) > 0, "DNA sequence cannot be empty"

    # embed protein sequences
    dna_embeddings = generate_dna_embeddings(
        model=model,
        tokenizer=tokenizer,
        dna_sequence=dna_sequence.upper(),
        model_type=model_type,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        dna_seq_overlap=dna_seq_overlap,
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

    return dna_embeddings
