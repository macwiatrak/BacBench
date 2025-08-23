import logging
from datetime import datetime
from typing import Any

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

try:
    import pyrodigal
except ImportError:
    logging.warning("pyrodigal is not installed. Please install it if you want to use gLM2.")


def extract_cds_and_intergenic_regions(
    sequences: list[str],
    contig_names: list[str] = None,
    output_filepath: str = None,  # optional: write GenBank
) -> pd.DataFrame:
    """
    Extract CDS and intergenic regions from a list of sequences.

    Returns
    -------
        pd.DataFrame with columns: sequence, start, end, strand, sequence_type
        (start/end are 1-based inclusive; strand is +1/-1 for CDS, 0 for intergenic)
    """
    if contig_names is None:
        contig_names = [f"contig_{i}" for i in range(len(sequences))]
    assert len(sequences) == len(contig_names), "Number of sequences and contig names must match."

    # normalize to uppercase strings for pyrodigal
    seqs = [str(s).upper() for s in sequences]

    # Train ORF finder on all contigs (varargs; pyrodigal will join with TTAATTAATTAA)
    joint_seq = "TTAATTAATTAA".join(seqs)  # pyrodigal uses this as a separator
    if len(seqs) == 0:
        raise ValueError("No sequences provided.")
    elif len(joint_seq) > 20000:
        orf_finder = pyrodigal.GeneFinder()
        orf_finder.train(joint_seq)
    else:
        orf_finder = pyrodigal.GeneFinder(meta=True)  # use meta mode for short sequences

    today_str = datetime.today().strftime("%d-%b-%Y").upper()

    records: list[SeqRecord] = []
    rows: list[dict[str, Any]] = []

    for contig_idx, (contig_id, seq_str) in enumerate(zip(contig_names, seqs, strict=False)):
        seq_len = len(seq_str)
        # Build record (Biopython prefers Seq)
        new_record = SeqRecord(
            Seq(seq_str),
            id=f"{contig_idx}_{contig_id}",
            name=contig_id,
            description=f"Pyrodigal results: {contig_id}",
            annotations={"molecule_type": "DNA", "date": today_str},
        )

        # Find ORFs on this contig
        orfs = orf_finder.find_genes(seq_str)

        # Collect CDS intervals (1-based inclusive) and features
        cds_intervals = []
        features: list[SeqFeature] = []

        # --- CDS Features ---
        for pred in orfs:
            start = pred.begin  # 1-based inclusive
            end = pred.end
            strand = 1 if pred.strand == 1 else -1

            location = FeatureLocation(start - 1, end, strand=strand)  # 0-based, half-open
            qualifiers = {
                # "feature_id": [f"{contig_id}_{feature_id}"],
                "translation": [str(pred.translate())],
            }

            features.append(SeqFeature(location=location, type="CDS", qualifiers=qualifiers))
            cds_intervals.append((start, end))

        # --- Intergenic Features ---
        cds_intervals.sort()
        last_end = 0
        for start, end in cds_intervals:
            if last_end + 1 < start:
                # intergenic region
                intergenic_start = last_end + 1
                intergenic_end = start - 1
                location = FeatureLocation(intergenic_start - 1, intergenic_end)  # 0-based
                qualifiers = {
                    # "feature_id": [f"{contig_id}_{feature_id}"],
                    # "note": ["intergenic region"],
                    "sequence": [str(seq_str[intergenic_start - 1 : intergenic_end])]
                }
                features.append(SeqFeature(location=location, type="intergenic", qualifiers=qualifiers))
            last_end = max(last_end, end)

        if last_end < seq_len:
            intergenic_start = last_end + 1
            intergenic_end = seq_len
            location = FeatureLocation(intergenic_start - 1, intergenic_end)
            qualifiers = {
                # "feature_id": [f"{contig_id}_{feature_id}"],
                # "note": ["intergenic region"],
                "sequence": [str(seq_str[intergenic_start - 1 : intergenic_end])]
            }
            features.append(SeqFeature(location=location, type="intergenic", qualifiers=qualifiers))

        # Sort features by genomic start
        new_record.features = sorted(features, key=lambda f: int(f.location.start))

        # add feature id in order
        feature_id = 1
        for feature in new_record.features:
            feature.qualifiers["feature_id"] = [f"{contig_id}_{feature_id}"]
            feature_id += 1

        # Assign feature IDs in order
        for i, feat in enumerate(new_record.features, start=1):
            feat.qualifiers.setdefault("feature_id", [f"{contig_id}_{i}"])

            # ---- Build DataFrame row ----
            start_0b = int(feat.location.start)
            end_0b_excl = int(feat.location.end)
            start_1b = start_0b + 1
            end_1b = end_0b_excl  # because FeatureLocation is half-open

            if feat.type.lower() == "cds":
                seq_val = feat.qualifiers.get("translation", [""])
                seq_val = seq_val[0] if isinstance(seq_val, list) else seq_val
                strand_val = int(feat.strand) if feat.strand in (-1, 1) else 0
                seq_type = "cds"
            else:  # intergenic
                seq_val = feat.qualifiers.get("sequence", [""])
                seq_val = seq_val[0] if isinstance(seq_val, list) else seq_val
                strand_val = 0
                seq_type = "intergenic"

            rows.append(
                {
                    "sequence": seq_val,
                    "start": start_1b,  # 1-based inclusive
                    "end": end_1b,  # 1-based inclusive
                    "strand": strand_val,  # +1/-1 for CDS, 0 for intergenic
                    "sequence_type": seq_type,  # "cds" | "intergenic"
                    # Optional but useful if you want it:
                    "contig_idx": contig_idx,
                    "contig_name": contig_id,
                }
            )

        records.append(new_record)

    # Optional: write GenBank
    if output_filepath:
        with open(output_filepath, "w") as out_handle:
            SeqIO.write(records, out_handle, "genbank")

    # Build and sort DataFrame
    df = pd.DataFrame(rows)
    df = df.sort_values(["contig_idx", "start"]).reset_index(drop=True)
    return df


def dna_seq_to_cds_and_intergenic(
    dna_sequences: list[str] | str,
    contig_names: list[str] = None,
    output_filepath: str = None,  # optional: write GenBank
) -> pd.DataFrame:
    """
    Convert DNA sequences to a DataFrame with CDS and intergenic regions.

    Args:
        dna_sequences (List[str]): List of DNA sequences.
        contig_names (List[str], optional): Names for each contig. Defaults to None.
        output_filepath (str, optional): If provided, write results to a GenBank file.

    Returns
    -------
        pd.DataFrame: DataFrame with columns: sequence, start, end, strand, sequence_type.
    """
    if isinstance(dna_sequences, str):
        # If a single string is provided, convert to a list by splitting
        dna_sequences = dna_sequences.split()
    return extract_cds_and_intergenic_regions(dna_sequences, contig_names, output_filepath)
