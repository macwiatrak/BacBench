import os

import pandas as pd
from tap import Tap
from tqdm import tqdm


def run(
    input_dir: str,
    output_filepath: str,
):
    """Run the collection of protein cluster indices."""
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")])

    out = []
    for f in tqdm(files):
        df = pd.read_parquet(os.path.join(input_dir, f), columns=["Genome Name", "start", "Protein Name", "indices"])
        df = df.explode(["start", "Protein Name", "indices"]).explode(["start", "Protein Name", "indices"])

        out.append(df)

    out = pd.concat(out, ignore_index=True)
    out.to_parquet(output_filepath, index=False)


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_dir: str = "/rds/user/mw896/rds-flotolab-9X9gY1OFt4M/projects/bacformer/input-data/complete_genomes/val"
    output_filepath: str


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_dir=args.input_dir,
        output_filepath=args.output_filepath,
    )
