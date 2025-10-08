import os

import pandas as pd
import torch
from datasets import tqdm
from tap import Tap

from bacbench.modeling.embedder import load_seq_embedder
from bacbench.modeling.utils.utils_glm2 import precompute_glm2_elements, preprocess_glm2_gene_seq


def run(
    input_dir: str,
    model_path: str,
    output_dir: str,
    max_seq_len: int = 4096,
    start_idx: int = None,
    end_idx: int = None,
):
    """
    Run the embedding of essential genes from a dataset.

    Args:
        input_dir (str): Directory containing input parquet files.
        model_path (str): Path to the model for embedding.
        output_dir (str): Directory to save the output files.
        max_seq_len (int): Maximum sequence length for the model.
        start_idx (int | None): Start index for processing files.
        end_idx (int | None): End index for processing files.
    """
    os.makedirs(output_dir, exist_ok=True)

    embedder = load_seq_embedder(model_path)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")])
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".parquet")]
    files = [f for f in files if f not in existing_files]
    start_idx = start_idx if start_idx is not None else 0
    end_idx = end_idx if end_idx is not None else len(files)
    files = files[start_idx:end_idx]
    print(f"Processing {len(files)} files, from {start_idx}, to {end_idx}.")

    for f in files:
        df = pd.read_parquet(os.path.join(input_dir, f))
        df = df.explode(
            [
                "contig_id",
                "gene_name",
                "locus_tag",
                "labels",
                "start",
                "end",
                "strand",
                "protein_id",
                "protein_sequence",
                "dna_sequence",
            ]
        )

        output = []
        for _, row in tqdm(df.iterrows()):
            elements, gene_idx_to_elem_idx = precompute_glm2_elements(
                prot_seqs=row["protein_sequence"],
                dna_seq=row["dna_sequence"],
                start=row["start"],
                end=row["end"],
                strand=row["strand"],
            )

            for gene_idx in range(len(row["start"])):
                label = row["labels"][gene_idx]
                if label is None:
                    continue
                seq_str, gene_mask = preprocess_glm2_gene_seq(
                    elements=elements,
                    gene_idx_to_elem_idx=gene_idx_to_elem_idx,
                    gene_idx=gene_idx,  # Assuming start is the gene index here
                    max_seq_len=max_seq_len,
                )
                with torch.no_grad():
                    dna_representations = embedder([seq_str], max_seq_len, pooling="mean", gene_mask=[gene_mask])
                output.append(
                    {
                        "strain_name": row["strain_name"],
                        "contig_id": row["contig_id"],
                        "labels": label,
                        "embeddings": dna_representations[0],
                    }
                )

        output = pd.DataFrame(output)

        output.to_parquet(
            os.path.join(output_dir, f),
            index=False,
        )


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # ──────────────────────────────────────────────────────────
    input_dir: str = "/projects/u5ah/public/benchmarks/tasks/prot-func/data"
    output_dir: str = "/projects/u5ah/public/benchmarks/tasks/prot-func/glm2"
    model_path: str = "tattabio/gLM2_650M"
    max_seq_len: int = 4096
    start_idx: int = None
    end_idx: int = None


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_dir=args.input_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
