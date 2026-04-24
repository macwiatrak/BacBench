import os

import pandas as pd
import pyarrow.parquet as pq
from tap import Tap
from tqdm import tqdm

from bacbench.pp import dna_seq_to_cds_and_intergenic


def run(
    input_parquet_path: str,
    output_filepath: str,
    batch_size: int = 10,
    min_seq_len: int = 3,
) -> pd.DataFrame:
    """Extract CDS and intergenic sequences from a parquet file."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    pf = pq.ParquetFile(input_parquet_path)

    out = []
    for idx, batch in enumerate(tqdm(pf.iter_batches(batch_size=batch_size), desc="Processing genomes")):
        df = batch.to_pandas()
        for _, example in df.iterrows():
            genome_df = dna_seq_to_cds_and_intergenic(example["dna_sequence"].split())
            genome_df["seq_len"] = genome_df["sequence"].apply(len)
            genome_df = genome_df[genome_df["seq_len"] >= min_seq_len]
            genome_df = genome_df[["sequence", "strand", "sequence_type"]].drop_duplicates("sequence")
            out.append(genome_df)

        if idx >= 10:  # for testing, remove this condition to process all batches
            break

    out = pd.concat(out, ignore_index=True).drop_duplicates("sequence").reset_index(drop=True)
    out.to_parquet(output_filepath)


class ArgumentParser(Tap):
    """Argument parser for whole-genome BacLM embedding."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_parquet_path: str = "/projects/public/u6fp/benchmarks/tasks/phenotypic-traits/pheno_all_genomes_dna.parquet"
    output_filepath: str
    batch_size: int = 10
    min_seq_len: int = 3


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_parquet_path=args.input_parquet_path,
        output_filepath=args.output_filepath,
        batch_size=args.batch_size,
        min_seq_len=args.min_seq_len,
    )
