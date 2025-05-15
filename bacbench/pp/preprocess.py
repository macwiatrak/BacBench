import gzip
import os
import tempfile
from contextlib import nullcontext
from typing import Literal

import pandas as pd
import requests
from Bio import SeqIO


def download_genome_assembly_by_taxid(
    taxid: int,
    file_type: Literal["gbff", "gff", "fna"] = "gbff",
    output_dir: str = None,
):
    """Search for assemblies by TaxID, pick the first (or best) assembly, then download the desired genomic data by file type.

    Args:
        taxid: int
            NCBI Taxonomy ID (e.g. 511145 for E. coli K-12)
        file_type: str
            File type to download, one of "gbff", "gff", "fna" (i.e. fasta)
        output_dir: str
            Output directory to save the downloaded file
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # 1) E-search: find assembly UIDs
    esearch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=assembly&term=txid{taxid}[Organism]&retmode=json"
    )
    resp = requests.get(esearch_url)
    resp.raise_for_status()
    data = resp.json()

    idlist = data["esearchresult"]["idlist"]
    if not idlist:
        print(f"No assemblies found for TaxID {taxid}")
        return
    # Pick the first assembly UID, though you may want a more complex selection logic
    asm_uid = idlist[0]
    print(f"Found assembly UID {asm_uid} for TaxID {taxid}")

    # 2) E-summary: get assembly metadata
    esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={asm_uid}&retmode=json"
    resp2 = requests.get(esummary_url)
    resp2.raise_for_status()
    data2 = resp2.json()

    docsum = data2["result"][asm_uid]
    # Get the GCF accession (RefSeq)
    accession = docsum["assemblyaccession"]  # e.g. "GCF_000123456.1"
    print(f"GCF accession: {accession}")

    # RefSeq FTP path (also available: ftppath_genbank)
    refseq_ftp = docsum["ftppath_refseq"]  # e.g. "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/..."

    if not refseq_ftp:
        print("No RefSeq FTP path found for this assembly.")
        return

    # 3) Construct the URL for the file
    # The last part of the ftp path is typically the same name as the file prefix
    # For example, if ftp_path is:
    #   ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/123/456/GCF_000123456.1_ASM12345v1
    # then the file prefix is "GCF_000123456.1_ASM12345v1"
    file_prefix = refseq_ftp.split("/")[-1]
    filename = file_prefix + f"_genomic.{file_type}.gz"

    # Use HTTPS instead of FTP to avoid potential FTP issues
    # Just replace "ftp://" with "https://", as NCBI supports both
    https_ftp = refseq_ftp.replace("ftp://", "https://")
    file_url = f"{https_ftp}/{filename}"
    print(f"Downloading {file_url} ...")

    # 4) Download the file
    r = requests.get(file_url, stream=True)
    r.raise_for_status()
    out_file = os.path.join(output_dir, f"{taxid}_{accession}_genomic.{file_type}.gz")
    with open(out_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded genome assembly to {out_file}")


def extract_protein_info_from_gbff(filepath: str) -> pd.DataFrame:
    """
    Extract CDS-level information (including strand) from a GenBank file.

    Parameters
    ----------
    filepath : str
        Path to a *.gbff* or *.gbff.gz* file.

    Returns
    -------
    pd.DataFrame
        Columns
        [strain_name, accession_id, gene_name, protein_name,
         start, end, strand, protein_id, contig_name, protein_sequence]
    """
    # ------------------------------------------------------------------ #
    # Open (gzip-compressed or plain) file                               #
    # ------------------------------------------------------------------ #
    handle = gzip.open(filepath, "rt") if filepath.endswith(".gz") else open(filepath)

    records = SeqIO.parse(handle, "genbank")
    data: list[dict] = []

    for record in records:
        accession_id = record.annotations.get("accessions", [None])[0]
        strain_name = record.annotations.get("organism", None)
        contig_name = record.id  # e.g. “NC_000913.3”

        for feature in record.features:
            if feature.type != "CDS":
                continue

            # Per-feature metadata
            gene_name = feature.qualifiers.get("gene", [None])[0]
            locus_tag = feature.qualifiers.get("locus_tag", [None])[0]
            old_locus_tag = feature.qualifiers.get("old_locus_tag", [None])[0]
            protein_id = feature.qualifiers.get("protein_id", [None])[0]
            translation = feature.qualifiers.get("translation", [None])[0]

            # Skip pseudo / partial CDSs without a translation
            if translation is None:
                continue

            # Strand: 1  → “+”,  -1 → “-”,  None → None
            # strand_symbol = {1: "+", -1: "-"}.get(feature.location.strand)

            data.append(
                {
                    "strain_name": strain_name,
                    "accession_id": accession_id,
                    "gene_name": gene_name,
                    "locus_tag": locus_tag,
                    "old_locus_tag": old_locus_tag,
                    "start": int(feature.location.start),
                    "end": int(feature.location.end),
                    "strand": feature.location.strand,
                    "protein_id": protein_id,
                    "contig_name": contig_name,
                    "protein_sequence": translation,
                }
            )

    handle.close()
    return pd.DataFrame(data)


def extract_protein_info_from_gff(filepath: str) -> pd.DataFrame:
    """Parse a RefSeq / GenBank *.gff* (or *.gff.gz*) genome file and return the CDS-level information.

    Columns
    -------
    [accession_id, gene_name, locus_tag, product,
     start, end, strand, protein_id, contig_name]
    """
    import gzip

    import pandas as pd

    strand_to_int = {"+": 1, "-": -1}.get

    accession_id: str | None = None
    rows: list[dict] = []

    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt") as fh:
        for line in fh:
            # ---------- header handling ---------------------------------
            if (
                line.startswith("#!genome-build-accession")
                or line.startswith("#!assembly-accession")
                or line.startswith("#!assembly_accession")
            ):
                # take the final whitespace-separated token
                token = line.strip().split()[-1]
                # strip any “NCBI_Assembly:” prefix if present
                accession_id = token.split(":")[-1]
                continue

            # ignore other comments / pragma lines
            if line.startswith("#"):
                continue

            # ---------- data lines --------------------------------------
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "CDS":
                continue  # keep only CDS features

            seqid, _, _, start, end, _, strand, _, attr = cols
            attrs = dict(kv.split("=", 1) for kv in attr.split(";") if "=" in kv)

            rows.append(
                {
                    "accession_id": accession_id,
                    "gene_name": attrs.get("gene"),
                    "locus_tag": attrs.get("locus_tag"),
                    "product": attrs.get("product"),
                    "start": int(start),
                    "end": int(end),
                    "strand": strand_to_int(strand),
                    "protein_id": attrs.get("protein_id"),
                    "contig_name": seqid,
                }
            )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# FASTA / *.fna* → contig-level DataFrame
# -------------------------------------------------------------------------
def extract_dna_info_from_fna(filepath: str) -> pd.DataFrame:
    """Parse an NCBI *.fna* (FASTA) genome file – plain or *.gz – and return basic contig-level information.

    Returns
    -------
    pd.DataFrame
        Columns
        [accession_id, contig_name, length, dna_sequence]
        *accession_id* is pulled from the FASTA header if present,
        otherwise None.
    """
    handle = gzip.open(filepath, "rt") if filepath.endswith(".gz") else open(filepath)
    records = SeqIO.parse(handle, "fasta")

    rows: list[dict] = []
    for rec in records:
        # Typical header forms:  "NC_000913.3 Escherichia coli ..."  OR
        #                        ">NW_0123456.1 ..."
        header_fields = rec.description.split()
        accession_id = header_fields[0] if header_fields else None

        rows.append(
            {
                "accession_id": accession_id,
                "contig_name": rec.id,
                "length": len(rec.seq),
                "dna_sequence": str(rec.seq),
            }
        )
    handle.close()
    return pd.DataFrame(rows)


def download_and_process_genome_by_taxid(
    taxid: int,
    file_type: Literal["gbff", "gff", "fna"] = "gbff",
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Download an NCBI genome assembly by TaxID, parse it, and return a DataFrame.

    Download an NCBI genome assembly for the given TaxID, parse it with
    `extract_protein_info_from_genbank`, and return the resulting DataFrame.
    If `output_dir` is None, a temporary directory is created and removed
    automatically once the function finishes.

    Returns
    -------
    pd.DataFrame
        Columns: [strain_name, accession_id, accession_name, gene_name,
                  protein_name, start, end, protein_id, contig_idx,
                  protein_sequence]
    """
    # ---------------------------------------------------------------------
    # Branch 1 ― Use a temporary, auto-cleaned workspace
    # ---------------------------------------------------------------------
    if output_dir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download
            download_genome_assembly_by_taxid(taxid, file_type=file_type, output_dir=tmpdir)

            # There is only one file in the tmpdir
            try:
                genome_file = next(f for f in os.listdir(tmpdir) if f.endswith(f".{file_type}.gz"))
            except StopIteration:
                raise FileNotFoundError("Download finished but no genome file was found in the temporary directory.")  # noqa

            genome_path = os.path.join(tmpdir, genome_file)

            # inside both the TemporaryDirectory branch and the explicit-dir branch
            if file_type == "gbff":
                df = extract_protein_info_from_gbff(genome_path)
            elif file_type == "gff":
                df = extract_protein_info_from_gff(genome_path)
            else:  # "fna"
                df = extract_dna_info_from_fna(genome_path)

            return df

    # ---------------------------------------------------------------------
    # Branch 2 ― Caller supplied an explicit output_dir
    # ---------------------------------------------------------------------
    download_genome_assembly_by_taxid(taxid, file_type=file_type, output_dir=output_dir)

    # Fall back on “newest matching file” to avoid clashing with old downloads
    matches = [
        f for f in os.listdir(output_dir) if f.startswith(f"{taxid}_") and f.endswith(f"_genomic.{file_type}.gz")
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not locate the downloaded file in {output_dir!r}. Check that the download completed successfully."
        )

    matches.sort(key=lambda fn: os.path.getmtime(os.path.join(output_dir, fn)), reverse=True)
    genome_path = os.path.join(output_dir, matches[0])

    # inside both the TemporaryDirectory branch and the explicit-dir branch
    if file_type == "gbff":
        df = extract_protein_info_from_gbff(genome_path)
    elif file_type == "gff":
        df = extract_protein_info_from_gff(genome_path)
    else:  # "fna"
        df = extract_dna_info_from_fna(genome_path)

    return df


# -------------------------------------------------------------------------
# 1)  Download + parse by *assembly* accession (e.g. GCA_000006765.1)
# -------------------------------------------------------------------------
def download_and_process_genome_by_assembly_id(
    assembly_id: str,
    file_type: Literal["gbff", "gff", "fna"] = "gbff",
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Download an NCBI genome assembly given an accession_id.

    Download an NCBI genome assembly given an accession (GCA_*/GCF_*) and
    return a CDS–level DataFrame identical to
    `extract_protein_info_from_genbank`.

    Parameters
    ----------
    assembly_id : str
        Assembly accession, e.g. "GCA_000006765.1" or "GCF_000005845.2".
    file_type : {"gbff","gff", "fna"}
        Which annotation file to retrieve.
    output_dir : str | None
        Where to save the .gz file; a TemporaryDirectory is created if None.

    Returns
    -------
    pd.DataFrame
        Same columns as produced by `extract_protein_info_from_genbank`.
    """
    if file_type not in {"gbff", "gff", "fna"}:
        raise ValueError("file_type must be 'gbff', 'gff' or 'fna'")

    def _download(acc: str, typ: str, dest: str) -> str:
        """Internal helper: grab <acc>_genomic.<typ>.gz to dest, return path"""
        # --- 1. translate accession → assembly UID -----------------------
        esearch = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=assembly&term={acc}[Assembly%20Accession]&retmode=json"
        )
        uid = requests.get(esearch).json()["esearchresult"]["idlist"]
        if not uid:
            raise ValueError(f"Assembly accession {acc} not found on NCBI")
        uid = uid[0]

        # --- 2. fetch metadata (ftppath_refseq | ftppath_genbank) --------
        esummary = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={uid}&retmode=json"
        doc = requests.get(esummary).json()["result"][uid]

        ftp_path = doc["ftppath_refseq"] or doc["ftppath_genbank"]
        if not ftp_path:
            raise RuntimeError("No FTP path found for this assembly.")

        prefix = ftp_path.split("/")[-1]
        fname = f"{prefix}_genomic.{typ}.gz"
        url = f"{ftp_path.replace('ftp://', 'https://')}/{fname}"

        file_out = os.path.join(dest, fname)
        with requests.get(url, stream=True) as r, open(file_out, "wb") as fh:
            r.raise_for_status()
            for chunk in r.iter_content(8192):
                fh.write(chunk)
        return file_out

    # ---- use a temp dir unless the user wants to keep the raw file -------
    with tempfile.TemporaryDirectory() if output_dir is None else nullcontext(output_dir) as workdir:
        genome_gz = _download(assembly_id, file_type, workdir)

        if file_type == "gbff":
            df = extract_protein_info_from_gbff(genome_gz)
        elif file_type == "gff":
            df = extract_protein_info_from_gff(genome_gz)
        else:  # "fna"
            df = extract_dna_info_from_fna(genome_gz)

        # returned DataFrame already matches the genbank schema
        return df
