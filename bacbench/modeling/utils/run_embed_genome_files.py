import os.path
from functools import partial

import pandas as pd
import torch
from tap import Tap
from tqdm import tqdm

from bacbench.modeling.embed_prot_seqs import generate_protein_embeddings
from bacbench.modeling.embedder import SeqEmbedder, load_seq_embedder

tqdm.pandas()


def compute_protein_embeddings(
    embedder: SeqEmbedder,
    batch_size: int = 64,
    max_seq_len: int = 1024,
    protein_sequences: list[str] = None,
):
    """Helper function to compute protein embeddings for a list of protein sequences."""
    df = pd.DataFrame({"protein_sequence": protein_sequences})
    df["seq_len"] = df["protein_sequence"].str.len()
    df["prot_idx"] = range(len(df))
    df = df.sort_values(by="seq_len", ascending=False)
    prot_embs = generate_protein_embeddings(
        embedder=embedder,
        protein_sequences=df["protein_sequence"].tolist(),
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    df["prot_embeddings"] = prot_embs
    df = df.sort_values(by="prot_idx", ascending=True)
    return df["prot_embeddings"].tolist()


def run(
    input_dir: str,
    output_dir: str,
    batch_size: int = 128,
    model_path: str = "facebook/esm2_t12_35M_UR50D",
    start_idx: int = None,
    end_idx: int = None,
    output_col: str = "esm2_embeddings",
    max_seq_len: int = 1024,
):
    """Run processing"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load the data
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")])
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(files)
    files = files[start_idx:end_idx]
    print(f"Processing {len(files)} files, from {start_idx}, to {end_idx}.")

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load sequence embedder
    embedder = load_seq_embedder(model_path, device=device)
    compute_protein_embeddings_fn = partial(compute_protein_embeddings, embedder, batch_size, max_seq_len)

    # for each file in the input_dir
    for f in tqdm(files):
        df = pd.read_parquet(os.path.join(input_dir, f))
        # if output_col in df.columns:
        #     continue
        df[output_col] = df["protein_sequence"].progress_apply(lambda x: compute_protein_embeddings_fn(x))

        # save the dataframe
        output_file = os.path.join(output_dir, f)
        df.to_parquet(output_file, engine="pyarrow")


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_dir: str
    output_dir: str
    batch_size: int = 128
    model_path: str = "facebook/esm2_t12_35M_UR50D"
    start_idx: int = None
    end_idx: int = None
    max_seq_len: int = 1024
    output_col: str = "esm2_embeddings"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        model_path=args.model_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        output_col=args.output_col,
        max_seq_len=args.max_seq_len,
    )
