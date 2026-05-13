import json

import pandas as pd
from bacbench.tasks.ppi.data_reader import _split_rows_from_parquet


def _write_ppi_fixture(tmp_path):
    input_filepath = tmp_path / "ppi.parquet"
    split_filepath = tmp_path / "split.json"
    rows = pd.DataFrame(
        {
            "strain_name": ["train_a", "val_a", "test_a", "unknown_a"],
            "labels": [[], [], [], []],
            "embeddings": [[], [], [], []],
        }
    )
    rows.to_parquet(input_filepath)
    split_filepath.write_text(
        json.dumps(
            {
                "train_a": "train",
                "val_a": "validation",
                "test_a": "test",
            }
        )
    )
    return input_filepath, split_filepath


def test_incremental_parquet_split_matches_in_memory_split(tmp_path):
    input_filepath, split_filepath = _write_ppi_fixture(tmp_path)

    in_memory_splits = _split_rows_from_parquet(str(input_filepath), str(split_filepath))
    incremental_splits = _split_rows_from_parquet(
        str(input_filepath),
        str(split_filepath),
        use_incremental_read=True,
    )

    assert [rows["strain_name"].tolist() for rows in in_memory_splits] == [
        rows["strain_name"].tolist() for rows in incremental_splits
    ]


def test_incremental_parquet_split_handles_empty_split(tmp_path):
    input_filepath = tmp_path / "ppi.parquet"
    split_filepath = tmp_path / "split.json"
    pd.DataFrame(
        {
            "strain_name": ["train_a"],
            "labels": [[]],
            "embeddings": [[]],
        }
    ).to_parquet(input_filepath)
    split_filepath.write_text(json.dumps({"train_a": "train"}))

    train_rows, val_rows, test_rows = _split_rows_from_parquet(
        str(input_filepath),
        str(split_filepath),
        use_incremental_read=True,
    )

    assert train_rows["strain_name"].tolist() == ["train_a"]
    assert val_rows.empty
    assert test_rows.empty
