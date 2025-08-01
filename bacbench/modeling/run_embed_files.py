import os.path

import pandas as pd
import pyarrow as pa
import torch
from datasets import tqdm
from tap import Tap

from bacbench.modeling.embedder import load_seq_embedder


def add_vector_column(
    mtx: torch.Tensor,
    df: pd.DataFrame,
    col: str = "embeddings",
):
    """Add a vector column to the dataframe."""
    # ✅ columnar, zero‑copy, float16‑safe
    N, D = mtx.shape
    mtx = mtx.flatten().numpy()
    dtype = pa.float16() if mtx.dtype == torch.float16 else pa.float32()
    vec_col = pa.FixedSizeListArray.from_arrays(pa.array(mtx, type=dtype), D)

    df[col] = pd.Series(vec_col, dtype=pd.ArrowDtype(vec_col.type))
    return df


def run(
    input_dir: str,
    output_dir: str,
    batch_size: int = 128,
    model_path: str = "esmc_300m",
    start_idx: int = None,
    end_idx: int = None,
    col: str = "embeddings",
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

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load sequence embedder
    embedder = load_seq_embedder(model_path, device=device)

    # for each file in the input_dir
    for f in tqdm(files):
        df = pd.read_parquet(os.path.join(input_dir, f))[:10000]

        # check if the dataframe has the embeddings column
        if "embeddings" in df.columns:
            print(f"Skipping {f}, already has embeddings.")
            continue

        # get the sequence column and sort by len to speed up
        # the processing
        df["seq_len"] = df["protein_sequence"].str.len()
        df["prot_idx"] = range(len(df))
        df = df.sort_values(by="seq_len", ascending=False)

        prot_seqs = df["protein_sequence"].tolist()

        # batch the sequences
        embeddings_arr = []
        for i in tqdm(range(0, len(prot_seqs), batch_size)):
            batch_seqs = prot_seqs[i : i + batch_size]
            # embed the sequences
            with torch.no_grad():
                # get the embeddings
                batch_embeddings = embedder(
                    sequences=batch_seqs,
                    max_seq_len=1024,
                    return_numpy=False,
                ).cpu()
            # append the embeddings to the list
            embeddings_arr.append(batch_embeddings)
        del prot_seqs
        # add the embeddings to the dataframe
        embeddings_arr = torch.cat(embeddings_arr, dim=0)
        df = add_vector_column(embeddings_arr, df, col=col)
        # sort the dataframe by the original order
        df = df.sort_values(by="prot_idx").drop(columns=["seq_len", "prot_idx"])
        # save the dataframe to the output_dir
        output_file = os.path.join(output_dir, f)
        df.to_parquet(output_file, engine="pyarrow")


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_dir: str = "/rds/user/mw896/rds-flotolab-9X9gY1OFt4M/projects/bacformer/input-data/complete_genomes/ncbi_dataset/preprocessed"
    output_dir: str
    batch_size: int = 128
    model_path: str = "esmc_300m"
    start_idx: int = None
    end_idx: int = None
    col: str = "embeddings"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        model_path=args.model_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
