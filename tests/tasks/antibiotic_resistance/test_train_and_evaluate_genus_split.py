import importlib

import numpy as np
import pandas as pd

genus_eval = importlib.import_module("bacbench.tasks.antibiotic_resistance.train_and_evaluate_genus_split")


def test_select_model_names_excludes_metadata_columns():
    columns = ["genome_name", "bacformer", "esm2", "species", "genus", "family"]

    assert genus_eval.select_model_names(columns) == ["bacformer", "esm2"]
    assert genus_eval.select_model_names(columns, ["esm2"]) == ["esm2"]


def test_split_by_genus_is_disjoint_and_changes_with_seed():
    groups = np.repeat([f"genus_{index}" for index in range(12)], 2)
    y = np.tile(np.array([0, 1], dtype=np.float32), 12)

    split_one = genus_eval.split_by_genus(y, groups, 0.5, 0.25, 0.25, seed=1)
    split_two = genus_eval.split_by_genus(y, groups, 0.5, 0.25, 0.25, seed=2)

    for split in (split_one, split_two):
        train_groups = set(groups[split.train])
        val_groups = set(groups[split.val])
        test_groups = set(groups[split.test])
        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)
        assert all(set(y[indices]) == {0, 1} for indices in (split.train, split.val, split.test))
    assert split_one.fingerprint != split_two.fingerprint


def test_prepare_drugs_applies_existing_support_thresholds():
    model_df = pd.DataFrame(
        {
            "genus": [f"genus_{index // 2}" for index in range(12)],
            "eligible": [0, 1] * 6,
            "too_rare": [0] * 11 + [1],
        }
    )

    drugs = genus_eval.prepare_drugs(
        model_df,
        ["eligible", "too_rare"],
        total_min_samples=10,
        min_class_samples=2,
    )

    assert list(drugs) == ["eligible"]
    assert set(drugs["eligible"].y) == {0, 1}


def test_select_best_learning_rate_uses_mean_across_drugs():
    tuning_results = pd.DataFrame(
        {
            "model_name": ["model"] * 6,
            "drug": ["a", "b"] * 3,
            "lr": [0.05, 0.05, 0.01, 0.01, 0.005, 0.005],
            "val_auprc": [0.9, 0.1, 0.7, 0.7, 0.6, 0.6],
            "skipped": [None] * 6,
        }
    )

    best_lr, summary = genus_eval.select_best_learning_rate(tuning_results)

    assert best_lr == 0.01
    assert summary.loc[summary["selected"], "lr"].tolist() == [0.01]
