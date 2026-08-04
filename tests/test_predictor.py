"""Unit tests for `Predictor._collapse_augmentations` (bug 2 in the review)."""
from quannet.predictor import Predictor

import numpy as np
import pytest


def test_no_augmentations_returns_input_unchanged():
    """When n_predictions == n_structures (predict mode default), pass through."""
    preds = np.array([1.0, 2.0, 3.0])
    out = Predictor._collapse_augmentations(preds, n_structures=3)
    np.testing.assert_array_equal(out, preds)


def test_three_augmentations_per_structure_means_correctly():
    """3 structures × 3 augmentations = 9 predictions → mean per structure."""
    preds = np.array(
        [
            1.0,
            2.0,
            3.0,  # structure 0 → mean 2.0
            10.0,
            20.0,
            30.0,  # structure 1 → mean 20.0
            -1.0,
            0.0,
            1.0,  # structure 2 → mean 0.0 (regression: negative inputs OK)
        ]
    )
    out = Predictor._collapse_augmentations(preds, n_structures=3)
    np.testing.assert_allclose(out, [2.0, 20.0, 0.0])


def test_arbitrary_augmentation_count_inferred_from_length():
    """Function should not require knowing num_augmentations explicitly."""
    n_structures, n_augs = 4, 5
    preds = np.arange(n_structures * n_augs, dtype=float)
    out = Predictor._collapse_augmentations(preds, n_structures=n_structures)
    expected = preds.reshape(n_structures, n_augs).mean(axis=1)
    np.testing.assert_allclose(out, expected)
    assert out.shape == (n_structures,)


def test_raises_when_not_a_multiple():
    """5 predictions for 2 structures is ambiguous."""
    preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match='cannot collapse augmentations'):
        Predictor._collapse_augmentations(preds, n_structures=2)


def test_raises_when_predictions_empty():
    """0 predictions for any n_structures is invalid (can't have 0 augs)."""
    with pytest.raises(ValueError, match='cannot collapse augmentations'):
        Predictor._collapse_augmentations(np.array([]), n_structures=3)
