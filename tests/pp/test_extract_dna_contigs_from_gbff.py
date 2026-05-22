import tarfile

from bacbench.pp.extract_dna_contigs_from_gbff import get_genome_name, write_dna_contig_fasta
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord


def test_write_dna_contig_fasta_from_tarred_genbank(tmp_path):
    records = [
        SeqRecord(
            Seq("AAACCCGGG"),
            id="contig_1",
            description="first contig",
            annotations={"molecule_type": "DNA"},
            features=[SeqFeature(FeatureLocation(0, 3), type="intergenic")],
        ),
        SeqRecord(
            Seq("TTTGGGAAA"),
            id="contig_2",
            description="second contig",
            annotations={"molecule_type": "DNA"},
            features=[SeqFeature(FeatureLocation(2, 6), type="CDS")],
        ),
    ]
    genbank_path = tmp_path / "genome_a.fna.gbff"
    SeqIO.write(records, genbank_path, "genbank")

    archive_path = tmp_path / "genome_a.fna.gbff.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(genbank_path, arcname=genbank_path.name)

    output_path = write_dna_contig_fasta(archive_path, tmp_path)
    output_records = list(SeqIO.parse(output_path, "fasta"))

    assert output_path == tmp_path / "genome_a.fasta"
    assert [record.id for record in output_records] == ["contig_1", "contig_2"]
    assert [str(record.seq) for record in output_records] == ["AAACCCGGG", "TTTGGGAAA"]


def test_get_genome_name_uses_text_before_first_dot():
    assert get_genome_name("MGYG000098655.fna.gbff.tar.gz") == "MGYG000098655"
