import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from collections import defaultdict
from scripts.thedataloader import ChestXrayDataset, split_data

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_labels.csv")
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "images")

SEED = 42
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1


@pytest.fixture(scope="module")
def dataset():
    return ChestXrayDataset(CSV_PATH, IMG_DIR)


@pytest.fixture(scope="module")
def splits(dataset):
    train_idx, val_idx, test_idx = split_data(dataset, SEED, VAL_SPLIT, TEST_SPLIT)
    return train_idx, val_idx, test_idx


@pytest.fixture(scope="module")
def patient_sets(dataset, splits):
    train_idx, val_idx, test_idx = splits
    pids = dataset.patient_ids
    train_pids = set(pids.iloc[train_idx])
    val_pids   = set(pids.iloc[val_idx])
    test_pids  = set(pids.iloc[test_idx])
    return train_pids, val_pids, test_pids


# ── Patient overlap ───────────────────────────────────────────────────────────

class TestPatientOverlap:
    def test_no_train_test_overlap(self, patient_sets):
        train_pids, _, test_pids = patient_sets
        overlap = train_pids & test_pids
        assert len(overlap) == 0, f"Patient IDs appear in both train and test: {overlap}"

    def test_no_train_val_overlap(self, patient_sets):
        train_pids, val_pids, _ = patient_sets
        overlap = train_pids & val_pids
        assert len(overlap) == 0, f"Patient IDs appear in both train and val: {overlap}"

    def test_no_val_test_overlap(self, patient_sets):
        _, val_pids, test_pids = patient_sets
        overlap = val_pids & test_pids
        assert len(overlap) == 0, f"Patient IDs appear in both val and test: {overlap}"

    def test_all_patients_accounted_for(self, dataset, patient_sets):
        all_pids = set(dataset.patient_ids)
        train_pids, val_pids, test_pids = patient_sets
        assert train_pids | val_pids | test_pids == all_pids, (
            "Some patients are missing from the splits"
        )

    def test_split_sizes_match_config(self, dataset, splits):
        train_idx, val_idx, test_idx = splits
        total = len(dataset)
        # Allow ±2% tolerance due to rounding at patient level
        assert abs(len(test_idx) / total - TEST_SPLIT) < 0.02, (
            f"Test split proportion {len(test_idx)/total:.3f} deviates from target {TEST_SPLIT}"
        )
        assert abs(len(val_idx) / total - VAL_SPLIT) < 0.02, (
            f"Val split proportion {len(val_idx)/total:.3f} deviates from target {VAL_SPLIT}"
        )


# ── Label distribution ────────────────────────────────────────────────────────

def _label_prevalence(dataset, indices):
    """Return per-label positive rate for a given set of row indices."""
    meta = dataset.metadata.iloc[indices]
    label_cols = dataset.labels
    return meta[label_cols].mean()  # fraction of positives per label


class TestLabelDistribution:
    def test_test_distribution_close_to_train(self, dataset, splits):
        """
        Each label's positive rate in the test set should be within 15 pp of
        the training set rate. Loose tolerance because the sample is small
        and splitting is patient-level (not stratified by label).
        """
        train_idx, _, test_idx = splits
        train_prev = _label_prevalence(dataset, train_idx)
        test_prev  = _label_prevalence(dataset, test_idx)
        max_diff = (train_prev - test_prev).abs().max()
        assert max_diff < 0.15, (
            f"Largest per-label prevalence gap between train and test is {max_diff:.3f} "
            f"(>= 0.15). Per-label diff:\n{(train_prev - test_prev).abs().sort_values(ascending=False)}"
        )

    def test_val_distribution_close_to_train(self, dataset, splits):
        train_idx, val_idx, _ = splits
        train_prev = _label_prevalence(dataset, train_idx)
        val_prev   = _label_prevalence(dataset, val_idx)
        max_diff = (train_prev - val_prev).abs().max()
        assert max_diff < 0.15, (
            f"Largest per-label prevalence gap between train and val is {max_diff:.3f} "
            f"(>= 0.15). Per-label diff:\n{(train_prev - val_prev).abs().sort_values(ascending=False)}"
        )

    def test_no_label_missing_in_test(self, dataset, splits):
        """Every label present in train should appear at least once in test."""
        train_idx, _, test_idx = splits
        train_prev = _label_prevalence(dataset, train_idx)
        test_prev  = _label_prevalence(dataset, test_idx)
        present_in_train = train_prev[train_prev > 0].index
        missing_in_test = [lbl for lbl in present_in_train if test_prev[lbl] == 0]
        assert len(missing_in_test) == 0, (
            f"Labels present in train but absent in test: {missing_in_test}"
        )
