import os

import pandas as pd
import pyarrow.parquet as pq
from tap import Tap
from tqdm import tqdm


def run(
    input_dir: str,
    input_dir_prot_indices: str,
    output_dir: str,
    max_n_genomes_per_chunk: int = 50,
):
    """Run the matching of protein indices to genome indices."""
    # load the protein indices
    train_prot_indices = pd.read_parquet(os.path.join(input_dir_prot_indices, "train_prot_cluster_indices.parquet"))
    val_prot_indices = pd.read_parquet(os.path.join(input_dir_prot_indices, "val_prot_cluster_indices.parquet"))
    test_prot_indices = pd.read_parquet(os.path.join(input_dir_prot_indices, "test_prot_cluster_indices.parquet"))
    train_prot_indices["split"] = "train"
    val_prot_indices["split"] = "val"
    test_prot_indices["split"] = "test"

    prot_indices = pd.concat([train_prot_indices, val_prot_indices, test_prot_indices], ignore_index=True)
    del train_prot_indices, val_prot_indices, test_prot_indices
    prot_indices = prot_indices.rename(
        columns={"indices": "prot_cluster_id", "Protein Name": "protein_id", "Genome Name": "genome_name_full"}
    )
    prot_indices["genome_name"] = prot_indices["genome_name_full"].apply(
        lambda x: x.split("_")[-1]
    )  # remove GCA_ or GCF_

    train_idx = 1
    val_idx = 1
    test_idx = 1
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")])
    for f in tqdm(files):
        df = pq.read_table(os.path.join(input_dir, f))  # preserves FixedSizeList
        df = df.to_pandas(types_mapper=pd.ArrowDtype)  # stay Arrow‑backed
        genome_names = df["genome_name"].unique().tolist()
        prot_indices_subset = prot_indices[prot_indices["genome_name"].isin(genome_names)]
        df = pd.merge(df, prot_indices_subset, on=["genome_name", "start", "protein_id"], how="left")
        df = df.drop(columns=["genome_name"]).rename(columns={"genome_name_full": "genome_name"})
        # print("df columns:", df.columns)
        # print("prot_indices_subset columns:", prot_indices_subset.columns)
        df["prot_cluster_id"] = df["prot_cluster_id"].fillna(-100)
        for split in df["split"].unique():
            if split == "train":
                df_split = df[df["split"] == split]
                genomes = df_split["genome_name"].unique().tolist()
                print(f"Processing {len(genomes)} genomes for split {split} in file {f}")
                if len(genomes) > max_n_genomes_per_chunk:
                    df_split[df_split.genome_name.isin(genomes[: len(genomes) // 2])].to_parquet(
                        os.path.join(output_dir, "train", f"train_chunk_idx_{train_idx}.parquet"), engine="pyarrow"
                    )
                    df_split[df_split.genome_name.isin(genomes[len(genomes) // 2 :])].to_parquet(
                        os.path.join(output_dir, "train", f"train_chunk_idx_{train_idx + 1}.parquet"), engine="pyarrow"
                    )
                    train_idx += 2
                else:
                    df_split.to_parquet(
                        os.path.join(output_dir, "train", f"train_chunk_idx_{train_idx}.parquet"), engine="pyarrow"
                    )
                    train_idx += 1
            elif split == "val":
                df_split = df[df["split"] == split]
                genomes = df_split["genome_name"].unique().tolist()
                print(f"Processing {len(genomes)} genomes for split {split} in file {f}")
                if len(genomes) > max_n_genomes_per_chunk:
                    df_split[df_split.genome_name.isin(genomes[: len(genomes) // 2])].to_parquet(
                        os.path.join(output_dir, "val", f"val_chunk_idx_{val_idx}.parquet"), engine="pyarrow"
                    )
                    df_split[df_split.genome_name.isin(genomes[len(genomes) // 2 :])].to_parquet(
                        os.path.join(output_dir, "val", f"val_chunk_idx_{val_idx + 1}.parquet"), engine="pyarrow"
                    )
                    val_idx += 2
                else:
                    df_split.to_parquet(
                        os.path.join(output_dir, "val", f"val_chunk_idx_{val_idx}.parquet"), engine="pyarrow"
                    )
                    val_idx += 1
            elif split == "test":
                df_split = df[df["split"] == split]
                genomes = df_split["genome_name"].unique().tolist()
                print(f"Processing {len(genomes)} genomes for split {split} in file {f}")
                if len(genomes) > max_n_genomes_per_chunk:
                    df_split[df_split.genome_name.isin(genomes[: len(genomes) // 2])].to_parquet(
                        os.path.join(output_dir, "test", f"test_chunk_idx_{test_idx}.parquet"), index=False
                    )
                    df_split[df_split.genome_name.isin(genomes[len(genomes) // 2 :])].to_parquet(
                        os.path.join(output_dir, "test", f"test_chunk_idx_{test_idx + 1}.parquet"), index=False
                    )
                    test_idx += 2
                else:
                    df_split.to_parquet(
                        os.path.join(output_dir, "test", f"test_chunk_idx_{test_idx}.parquet"), index=False
                    )
                    test_idx += 1
        del df, prot_indices_subset


class ArgumentParser(Tap):
    """Argument parser for embedding protein sequences."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_dir: str = "/rds/user/mw896/rds-flotolab-9X9gY1OFt4M/projects/bacformer/input-data/complete_genomes/ncbi_dataset/preprocessed-esmc"
    input_dir_prot_indices: str = (
        "/rds/user/mw896/rds-flotolab-9X9gY1OFt4M/projects/bacformer/input-data/complete_genomes"
    )
    output_dir: str = (
        "/rds/user/mw896/rds-flotolab-9X9gY1OFt4M/projects/bacformer/input-data/complete_genomes/esmc-dataset"
    )
    max_n_genomes_per_chunk: int = 50


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        input_dir=args.input_dir,
        input_dir_prot_indices=args.input_dir_prot_indices,
        output_dir=args.output_dir,
        max_n_genomes_per_chunk=args.max_n_genomes_per_chunk,
    )
