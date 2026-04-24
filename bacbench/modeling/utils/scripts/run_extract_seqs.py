import os
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tap import Tap
from tqdm import tqdm

from bacbench.pp import dna_seq_to_cds_and_intergenic


def _get_contig_sequences(dna_sequence: list[str] | str) -> list[str]:
    """Normalise the dataset DNA field to a list of contig strings."""
    if isinstance(dna_sequence, str):
        return dna_sequence.split()
    return dna_sequence


def _flush_unique_rows(
    rows: list[dict[str, Any]],
    output_dir: str,
    chunk_idx: int,
    final_writer: pq.ParquetWriter | None,
) -> tuple[int, pq.ParquetWriter | None]:
    """Write one chunk of globally unique rows and append it to final.parquet."""
    if not rows:
        return chunk_idx, final_writer

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, os.path.join(output_dir, f"chunk_{chunk_idx}.parquet"))

    if final_writer is None:
        final_writer = pq.ParquetWriter(os.path.join(output_dir, "final.parquet"), table.schema)
    final_writer.write_table(table)

    rows.clear()
    return chunk_idx + 1, final_writer


def run(
    input_parquet_path: str,
    output_dir: str,
    batch_size: int = 10,
    min_seq_len: int = 3,
    max_batches: int | None = None,
) -> None:
    """Extract CDS and intergenic sequences from a parquet file with global exact dedup."""
    os.makedirs(output_dir, exist_ok=True)
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")

    pf = pq.ParquetFile(input_parquet_path)

    seen_sequences: set[str] = set()
    out: list[dict[str, Any]] = []
    chunk_idx = 0
    final_writer: pq.ParquetWriter | None = None

    try:
        for idx, batch in enumerate(tqdm(pf.iter_batches(batch_size=batch_size), desc="Processing genomes")):
            if max_batches is not None and idx >= max_batches:
                print(f"Reached max_batches={max_batches}, stopping early.")
                break

            for example in batch.to_pylist():
                genome_df = dna_seq_to_cds_and_intergenic(_get_contig_sequences(example["dna_sequence"]))
                genome_df["seq_len"] = genome_df["sequence"].apply(len)
                genome_df = genome_df.loc[genome_df["seq_len"] >= min_seq_len, ["sequence", "seq_len", "sequence_type"]]

                for sequence, seq_len, sequence_type in genome_df.itertuples(index=False, name=None):
                    if sequence in seen_sequences:
                        continue
                    seen_sequences.add(sequence)
                    out.append({"sequence": sequence, "seq_len": seq_len, "sequence_type": sequence_type})

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} batches, saving intermediate results...")
                chunk_idx, final_writer = _flush_unique_rows(out, output_dir, chunk_idx, final_writer)

        chunk_idx, final_writer = _flush_unique_rows(out, output_dir, chunk_idx, final_writer)
    finally:
        if final_writer is not None:
            final_writer.close()

    print(f"Saved {len(seen_sequences):,} globally unique sequences to {output_dir}")


class ArgumentParser(Tap):
    """Argument parser for whole-genome BacLM embedding."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_parquet_path: str = "/projects/public/u6fp/benchmarks/tasks/phenotypic-traits/pheno_all_genomes_dna.parquet"
    output_dir: str
    batch_size: int = 10
    min_seq_len: int = 3
    max_batches: int | None = None


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_parquet_path=args.input_parquet_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        min_seq_len=args.min_seq_len,
        max_batches=args.max_batches,
    )
