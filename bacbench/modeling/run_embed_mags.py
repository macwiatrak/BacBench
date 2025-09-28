import gzip
import io
import os
import random
import tarfile
from collections.abc import Iterable

import pandas as pd
import torch
from Bio import SeqIO
from tap import Tap
from tqdm import tqdm

from bacbench.modeling.embed_prot_seqs import generate_protein_embeddings
from bacbench.modeling.embedder import load_seq_embedder


def _iter_gbff_records_from_stream(text_stream) -> Iterable:
    """Yield SeqRecord objects from a text-mode file-like object containing GenBank text."""
    yield from SeqIO.parse(text_stream, "genbank")


def _iter_records_from_tar(tar_path: str) -> Iterable:
    """
    Yield SeqRecords from every .gbff/.gbk (optionally gz) entry in a .tar or .tar.gz archive.
    Intergenic regions are ignored upstream by only emitting records; filtering happens later.
    """
    mode = "r:gz" if tar_path.lower().endswith(".gz") else "r:"
    with tarfile.open(tar_path, mode) as tf:
        for m in tf.getmembers():
            name_lower = m.name.lower()
            if not any(name_lower.endswith(ext) for ext in (".gbff", ".gbk", ".gbff.gz", ".gbk.gz")):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            if name_lower.endswith(".gz"):
                with gzip.open(f, "rt") as gzf:
                    yield from _iter_gbff_records_from_stream(gzf)
            else:
                with io.TextIOWrapper(f) as txt:
                    yield from _iter_gbff_records_from_stream(txt)


def _iter_records_from_path(path: str) -> Iterable:
    """Yield SeqRecords from a path that may be a plain GenBank file, gz file, or tar/tar.gz archive."""
    lower = path.lower()
    if lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        yield from _iter_records_from_tar(path)
    elif lower.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            yield from _iter_gbff_records_from_stream(fh)
    else:
        with open(path) as fh:
            yield from _iter_gbff_records_from_stream(fh)


def extract_protein_info_from_gbff_any(filepath: str) -> pd.DataFrame:
    """
    Extract CDS-level information (including strand) from a GenBank file or a tar(.gz) archive
    of GenBank files. Intergenic regions are ignored (only CDS with a translation are returned).

    Parameters
    ----------
    filepath : str
        Path to a *.gbff* / *.gbk* (optionally *.gz*) file, or a .tar/.tar.gz archive
        containing such files.

    Returns
    -------
    pd.DataFrame
        Columns
        [strain_name, accession_id, gene_name, protein_name,
         start, end, strand, protein_id, contig_name, protein_sequence,
         locus_tag, old_locus_tag]
    """
    data: list[dict] = []

    for record in _iter_records_from_path(filepath):
        # Record-level metadata
        accessions = record.annotations.get("accessions")
        accession_id: str | None = accessions[0] if isinstance(accessions, list) and accessions else None
        contig_name = getattr(record, "id", None)  # e.g., "NC_000913.3"

        # Feature-level extraction (CDS only, ignoring intergenic)
        for feature in record.features:
            if feature.type != "CDS":
                continue

            translation = feature.qualifiers.get("translation", [None])[0]

            # Skip pseudo/partial CDSs without a translation
            if translation is None:
                continue

            data.append(
                {
                    "accession_id": accession_id,
                    "start": int(feature.location.start),
                    "end": int(feature.location.end),
                    "strand": feature.location.strand,  # 1 → plus, -1 → minus, None → unknown
                    "contig_name": contig_name,
                    "protein_sequence": translation,
                }
            )

    # Preserve column order (including locus_tag fields present in your original dict)
    return pd.DataFrame(
        data,
        columns=[
            "accession_id",
            "start",
            "end",
            "strand",
            "contig_name",
            "protein_sequence",
        ],
    )


def extract_all_filepaths(input_dir: str) -> list[str]:
    """
    Recursively find all .gbff/.gbk/.tar/.tar.gz files in the input directory.

    Parameters
    ----------
    input_dir : str
        The root directory to search for files.

    Returns
    -------
    List[str]
        A list of file paths to the found files.
    """
    import os

    filepaths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".gbff", ".gbk", ".tar", ".tar.gz", ".tgz", ".gz")):
                filepaths.append(os.path.join(root, file))

    return filepaths


def run(
    input_dir: str,
    output_dir: str,
    start_idx: int = None,
    end_idx: int = None,
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    chunk_size: int = 50,
    genome_metadata_fp: str = None,
):
    """Run the embedding process for all genomes in the input directory."""
    os.makedirs(output_dir, exist_ok=True)
    filepaths = extract_all_filepaths(input_dir)
    if genome_metadata_fp is not None:
        metadata_df = pd.read_csv(genome_metadata_fp, sep="\t")
        valid_genome_ids = set(metadata_df["genome_id"].tolist())
        filepaths = [f for f in filepaths if os.path.basename(f).split(".")[0] in valid_genome_ids]
    print(f"Found {len(filepaths)} files in {input_dir}")
    random.seed(1)
    random.shuffle(filepaths)

    start_idx = start_idx if start_idx is not None else 0
    end_idx = end_idx if end_idx is not None else len(filepaths)
    filepaths = filepaths[start_idx:end_idx]
    print(f"Processing files from index {start_idx} to {end_idx}, total {len(filepaths)} files")

    # create SeqEmbedder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = load_seq_embedder("Synthyra/ESMplusplus_small", device=device)

    output = []
    chunk_idx = 0
    for f in tqdm(filepaths):
        try:
            prot_seqs_df = extract_protein_info_from_gbff_any(f)
            prot_seqs_df["protein_sequence"] = prot_seqs_df["protein_sequence"].apply(lambda x: x.replace("*", ""))
            if prot_seqs_df.empty:
                print(f"No CDS features found in {f}, skipping.")
                continue
        except Exception as e:  # noqa
            print(f"Error processing {f}: {e}")
            continue
        # embed the genome using ESMC
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

        # if we pool at protein level, we need to return the embeddings in the same order as the input
        prot_seqs_df["protein_embedding"] = protein_embeddings
        # sort by protein index
        prot_seqs_df = prot_seqs_df.sort_values(by="protein_index")
        prot_seqs_df["genome_id"] = os.path.basename(f).split(".")[0]
        prot_seqs_df = (
            prot_seqs_df.groupby("genome_id").agg(list).reset_index().drop(columns=["protein_index", "prot_len"])
        )
        output.append(prot_seqs_df)

        if len(output) >= chunk_size:
            output_df = pd.concat(output, ignore_index=True)
            chunk_file = os.path.join(output_dir, f"embeddings_{start_idx}_{end_idx}_{chunk_idx}.parquet")
            output_df.to_parquet(chunk_file, engine="pyarrow")
            chunk_idx += 1
            output = []

    if len(output) > 0:
        output_df = pd.concat(output, ignore_index=True)
        chunk_file = os.path.join(output_dir, f"embeddings_{start_idx}_{end_idx}_{chunk_idx}.parquet")
        output_df.to_parquet(chunk_file, engine="pyarrow")


class ArgumentParser(Tap):
    """Argument parser for finetuning linear model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_dir: str
    output_dir: str
    start_idx: int = None
    end_idx: int = None
    batch_size: int = 64
    max_prot_seq_len: int = 1024
    chunk_size: int = 50
    genome_metadata_fp: str = None


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size,
        max_prot_seq_len=args.max_prot_seq_len,
        chunk_size=args.chunk_size,
        genome_metadata_fp=args.genome_metadata_fp,
    )
