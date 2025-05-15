import numpy as np
import torch

BACFORMER_SPECIAL_TOKENS_DICT = {
    "PAD": 0,
    "MASK": 1,
    "CLS": 2,
    "SEP": 3,
    "PROT_EMB": 4,
    "END": 5,
}


def get_prot_seq_col_name(cols: list[str]) -> str:
    """Get the protein sequence column name from the dataframe columns.

    Args:
        cols (List[str]): The list of column names.

    Returns
    -------
        str: The protein sequence column name.
    """
    if "protein_sequence" in cols:
        return "protein_sequence"
    if "protein_sequences" in cols:
        return "protein_sequences"
    if "sequence" in cols:
        return "sequence"
    raise ValueError("No protein sequence column found in the dataframe.")


def get_dna_seq_col_name(cols: list[str]) -> str:
    """Get the DNA sequence column name from the dataframe columns.

    Args:
        cols (List[str]): The list of column names.

    Returns
    -------
        str: The protein sequence column name.
    """
    if "sequence" in cols:
        return "sequence"
    if "dna_seq" in cols:
        return "dna_seq"
    if "dna_sequence" in cols:
        return "dna_sequence"
    raise ValueError("No DNA sequence column found in the dataframe.")


def protein_embeddings_to_inputs(
    protein_embeddings: list[list[np.ndarray]] | list[np.ndarray],
    max_n_proteins: int = 6000,
    max_n_contigs: int = 1000,
    cls_token_id: int = BACFORMER_SPECIAL_TOKENS_DICT["CLS"],
    sep_token_id: int = BACFORMER_SPECIAL_TOKENS_DICT["CLS"],
    prot_emb_token_id: int = BACFORMER_SPECIAL_TOKENS_DICT["PROT_EMB"],
    end_token_id: int = BACFORMER_SPECIAL_TOKENS_DICT["END"],
    torch_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convert protein embeddings to inputs for Bacformer model.

    Args:
        protein_embeddings (List[List[np.ndarray]]): The protein embeddings to convert.
        max_n_proteins (int): The maximum number of proteins to use for each genome.
        max_n_contigs (int): The maximum number of contigs to use for each genome.
        cls_token_id (int): The ID of the CLS token.
        sep_token_id (int): The ID of the SEP token.
        prot_emb_token_id (int): The ID of the protein embedding token.
        end_token_id (int): The ID of the END token.

    Returns
    -------
        dict: The inputs for the Bacformer model.
    """
    assert len(protein_embeddings) > 0, "protein_embeddings should not be empty"

    # check if protein_embeddings is a list of lists, if not, make it one
    if not isinstance(protein_embeddings[0], list):
        protein_embeddings = [protein_embeddings]

    # preprocess protein embeddings
    dim = len(protein_embeddings[0][0])
    pad_emb = torch.zeros(dim, dtype=torch_dtype)

    special_tokens_mask = [cls_token_id]
    protein_embeddings_output = [pad_emb]
    token_type_ids = [0]

    # iterate through contigs
    for contig_idx, contig in enumerate(protein_embeddings):
        # check if contig does not exceed max_n_contigs
        if contig_idx > max_n_contigs:
            contig_idx = max_n_contigs - 1
        # iterate through prots in contig
        for prot_emb in contig:
            # append prot_emb_token_id to special tokens mask
            special_tokens_mask.append(prot_emb_token_id)
            # append prot_emb to protein_embeddings_output
            protein_embeddings_output.append(torch.tensor(prot_emb, dtype=torch_dtype))
            # append contig_idx to token_type_ids
            token_type_ids.append(contig_idx)
        # separate the contigs with a SEP token
        special_tokens_mask.append(sep_token_id)
        protein_embeddings_output.append(pad_emb)
        token_type_ids.append(contig_idx)

    # account for the END token
    protein_embeddings_output = protein_embeddings_output[: max_n_proteins - 1] + [pad_emb]
    protein_embeddings_output = torch.stack(protein_embeddings_output)

    special_tokens_mask = special_tokens_mask[: max_n_proteins - 1] + [end_token_id]
    special_tokens_mask = torch.tensor(special_tokens_mask)

    token_type_ids = token_type_ids[: max_n_proteins - 1] + [contig_idx]
    token_type_ids = torch.tensor(token_type_ids)

    attention_mask = torch.ones_like(special_tokens_mask)
    return {
        "protein_embeddings": protein_embeddings_output.unsqueeze(0),
        "special_tokens_mask": special_tokens_mask.unsqueeze(0),
        "token_type_ids": token_type_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
    }
