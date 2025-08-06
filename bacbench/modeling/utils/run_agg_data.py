import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix column types in the dataframe."""
    for col in df.columns:
        if "bool" in str(df[col].dtype):
            df[col] = df[col].astype("boolean")
        elif "string" in str(df[col].dtype):
            df[col] = df[col].astype("string")
        elif "int" in str(df[col].dtype):
            df[col] = df[col].astype("int64")
        elif "double" in str(df[col].dtype):
            df[col] = df[col].astype("float64")
        elif "fixed_size_list" in str(df[col].dtype):
            df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float32))
        else:
            raise ValueError(f"Unsupported column type: {df[col].dtype}")
    # fix specific column
    df["prot_cluster_id"] = df["prot_cluster_id"].astype("int")
    return df


def run(
    input_dir: str,
):
    """Run aggregate data at genome-level for Bacformer 2."""
    for split in ["test", "val", "train"]:
        input_split_dir = os.path.join(input_dir, split)
        files = sorted([i for i in os.listdir(input_split_dir) if i.endswith(".parquet")])

        for f in tqdm(files):
            tbl = pq.read_table(os.path.join(input_split_dir, f))
            df = tbl.to_pandas(types_mapper=pd.ArrowDtype)

            df = fix_columns(df)
            # aggregate data at genome-level
            df = df.groupby("genome_name").agg(list).drop(columns=["split"]).reset_index(drop=False)
            # save the data
            df.to_parquet(os.path.join(input_split_dir, f))
            print("Finished processing file:", f)
            break


if __name__ == "__main__":
    input_dir = "/rds/user/mw896/rds-flotolab-9X9gY1OFt4M/projects/bacformer/input-data/complete_genomes/esmc-dataset"
    run(input_dir=input_dir)
