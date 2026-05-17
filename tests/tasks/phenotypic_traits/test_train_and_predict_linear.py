import importlib
import subprocess
import sys

import numpy as np
import pytest

PANDAS_IMPORTABLE = (
    subprocess.run(
        [sys.executable, "-c", "import pandas"],
        capture_output=True,
        check=False,
    ).returncode
    == 0
)

pytestmark = pytest.mark.skipif(not PANDAS_IMPORTABLE, reason="pandas is not importable in this environment")


def test_split_indices_uses_group_split_without_group_leakage():
    linear = importlib.import_module("bacbench.tasks.phenotypic_traits.train_and_predict_linear")
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array(["g1", "g1", "g2", "g2", "g3", "g3"])

    train_idx, val_idx, test_idx = linear._split_indices(
        y=y,
        groups=groups,
        split_mode="genus",
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        seed=1,
    )

    observed = set(train_idx) | set(val_idx) | set(test_idx)
    assert observed == set(range(len(y)))
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_run_infers_phenotype_columns_without_column_position_assumptions(monkeypatch):
    pd = importlib.import_module("pandas")
    linear = importlib.import_module("bacbench.tasks.phenotypic_traits.train_and_predict_linear")
    df = pd.DataFrame(
        {
            "genome_name": [f"genome_{i}" for i in range(6)],
            "embeddings": [np.array([float(i), float(i + 1)], dtype=np.float32) for i in range(6)],
            "genus": ["a", "a", "b", "b", "c", "c"],
            "species": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "trait_good": ["yes", "yes", "yes", "no", "no", "no"],
            "trait_rare": ["yes", "yes", "yes", "yes", "yes", "no"],
        }
    )
    calls = []

    def fake_train_and_predict(filtered_df, **kwargs):
        calls.append({"filtered_df": filtered_df, **kwargs})
        return {
            "phenotype": kwargs["phenotype"],
            "seed": kwargs["seed"],
            "embeddings_col": kwargs["embeddings_col"],
            "model_name": kwargs["model_name"],
            "split": kwargs["split"],
            "val_macro_auroc": 1.0,
            "val_macro_auprc": 1.0,
            "val_macro_f1": 1.0,
            "val_macro_accuracy": 1.0,
            "val_accuracy": 1.0,
        }

    monkeypatch.setattr(linear, "train_and_predict", fake_train_and_predict)

    result = linear.run(
        df=df,
        lr=0.01,
        embeddings_col="embeddings",
        split="genus",
        min_class_samples=2,
        seeds=[123],
        model_name="dummy_model",
    )

    assert result["phenotype"].tolist() == ["trait_good"]
    assert len(calls) == 1
    assert calls[0]["lr"] == 0.01
    assert calls[0]["embeddings_col"] == "embeddings"
    assert calls[0]["phenotype"] == "trait_good"
    assert calls[0]["max_epochs"] == 100
    assert calls[0]["early_stopping_patience"] == 10
    assert calls[0]["split"] == "genus"
    assert calls[0]["train_size"] == 0.7
    assert calls[0]["val_size"] == 0.1
    assert calls[0]["test_size"] == 0.2
    assert calls[0]["test_after_train"] is False
    assert calls[0]["model_name"] == "dummy_model"
    assert calls[0]["seed"] == 123
    assert "trait_rare" not in calls[0]["filtered_df"].columns
    assert {"genome_name", "embeddings", "genus", "trait_good"}.issubset(calls[0]["filtered_df"].columns)
