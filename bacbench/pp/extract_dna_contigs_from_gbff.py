from __future__ import annotations

import argparse
import gzip
import tarfile
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def get_genome_name(input_filepath: str | Path) -> str:
    """Extract the genome name from an input filename."""
    return Path(input_filepath).name.split(".", 1)[0]


@contextmanager
def open_genbank_text(input_filepath: str | Path):
    """Open a plain, gzipped, or tarred GenBank file as a text handle."""
    input_filepath = Path(input_filepath)

    if input_filepath.suffixes[-2:] == [".tar", ".gz"] or input_filepath.suffix == ".tgz":
        with tarfile.open(input_filepath, "r:gz") as archive:
            gbff_members = [
                member
                for member in archive.getmembers()
                if member.isfile() and (member.name.endswith(".gbff") or member.name.endswith(".gbk"))
            ]
            if not gbff_members:
                raise ValueError(f"No .gbff or .gbk file found in {input_filepath}.")
            if len(gbff_members) > 1:
                member_names = [member.name for member in gbff_members]
                raise ValueError(f"Expected one GenBank file in {input_filepath}, found {member_names}.")

            extracted = archive.extractfile(gbff_members[0])
            if extracted is None:
                raise ValueError(f"Could not read {gbff_members[0].name} from {input_filepath}.")

            with extracted, TextIOWrapper(extracted) as handle:
                yield handle
        return

    if input_filepath.suffix == ".gz":
        with gzip.open(input_filepath, "rt") as handle:
            yield handle
        return

    with input_filepath.open() as handle:
        yield handle


def genbank_dna_records(input_filepath: str | Path) -> list[SeqRecord]:
    """Read full DNA records from GenBank, ignoring CDS/intergenic features."""
    with open_genbank_text(input_filepath) as handle:
        return [record for record in SeqIO.parse(handle, "genbank") if len(record.seq) > 0]


def write_dna_contig_fasta(input_filepath: str | Path, output_dir: str | Path) -> Path:
    """Write one FASTA record per GenBank contig and return the output path."""
    genome_name = get_genome_name(input_filepath)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / f"{genome_name}.fasta"

    records = genbank_dna_records(input_filepath)
    if not records:
        raise ValueError(f"No DNA records found in {input_filepath}.")

    SeqIO.write(records, output_filepath, "fasta")
    return output_filepath


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract whole-contig DNA sequences from a GenBank file to FASTA.")
    parser.add_argument(
        "--input-filepath", required=True, help="Input .gbff, .gbff.gz, or .tar.gz containing one .gbff."
    )
    parser.add_argument("--output-dir", required=True, help="Directory where <genome_name>.fasta will be written.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_filepath = write_dna_contig_fasta(args.input_filepath, args.output_dir)
    print(output_filepath)
