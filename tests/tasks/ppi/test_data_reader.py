import pandas as pd
from bacbench.tasks.ppi.data_reader import _split_rows_from_parquet


def _write_ppi_fixture(tmp_path):
    input_filepath = tmp_path / "ppi.parquet"
    rows = pd.DataFrame(
        {
            "strain_name": ["train_a", "val_a", "test_a"],
            "split": ["train", "validation", "test"],
            "labels": [[], [], []],
            "embeddings": [[], [], []],
        }
    )
    rows.to_parquet(input_filepath)
    return input_filepath


def test_incremental_parquet_split_matches_in_memory_split(tmp_path):
    input_filepath = _write_ppi_fixture(tmp_path)

    in_memory_splits = _split_rows_from_parquet(str(input_filepath))
    incremental_splits = _split_rows_from_parquet(
        str(input_filepath),
        use_incremental_read=True,
    )

    assert [rows["strain_name"].tolist() for rows in in_memory_splits] == [
        rows["strain_name"].tolist() for rows in incremental_splits
    ]


def test_incremental_parquet_split_handles_empty_split(tmp_path):
    input_filepath = tmp_path / "ppi.parquet"
    pd.DataFrame(
        {
            "strain_name": ["train_a"],
            "split": ["train"],
            "labels": [[]],
            "embeddings": [[]],
        }
    ).to_parquet(input_filepath)

    train_rows, val_rows, test_rows = _split_rows_from_parquet(
        str(input_filepath),
        use_incremental_read=True,
    )

    assert train_rows["strain_name"].tolist() == ["train_a"]
    assert val_rows.empty
    assert test_rows.empty
