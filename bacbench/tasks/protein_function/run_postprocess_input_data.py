import json
import os

import pandas as pd
from tap import Tap
from tqdm import tqdm


def run(
    input_dir: str,
    strain2split_filepath: str,
):
    """Run the postprocessing of input data for protein function prediction."""
    with open(strain2split_filepath) as f:
        strain2split = json.load(f)

    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]

    train_df = []
    val_df = []
    test_df = []
    for f in tqdm(files):
        df = pd.read_parquet(os.path.join(input_dir, f))
        df = df.explode(
            ["contig_id", "gene_name", "locus_tag", "labels", "start", "end", "strand", "protein_id", "embeddings"]
        )
        df = df.explode(["gene_name", "locus_tag", "labels", "start", "end", "strand", "protein_id", "embeddings"])
        df = df.dropna(subset=["labels"]).rename(columns={"labels": "label"})
        df["split"] = df["strain_name"].map(strain2split)
        train_df.append(df[df["split"] == "train"])
        val_df.append(df[df["split"] == "val"])
        test_df.append(df[df["split"] == "test"])

    train_df = pd.concat(train_df, ignore_index=True)
    val_df = pd.concat(val_df, ignore_index=True)
    test_df = pd.concat(test_df, ignore_index=True)

    train_df.to_parquet(os.path.join(input_dir, "train.parquet"))
    val_df.to_parquet(os.path.join(input_dir, "val.parquet"))
    test_df.to_parquet(os.path.join(input_dir, "test.parquet"))


class ArgumentParser(Tap):
    """Argument parser for running data postprocessing."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # ──────────────────────────────────────────────────────────
    input_dir: str = "/projects/u5ah/public/benchmarks/tasks/prot-func/esm2"
    strain2split_filepath: str = "/projects/u5ah/public/benchmarks/tasks/prot-func/strain2split.json"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_dir=args.input_dir,
        strain2split_filepath=args.strain2split_filepath,
    )
