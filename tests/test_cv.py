from pathlib import Path

import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quannet.cv import composition, fold_indices, assign_lineage, lineage_stratified_folds  # noqa: E402

# Mirrors the 56-entity dataset composition (R1=17, R2=22, AB-001, TGN1412, 15 mAbs).
ENTITIES = (
    [f'R1-{i:03d}' for i in range(2, 19)]  # 17
    + [f'R2-{i:03d}' for i in range(1, 23)]  # 22
    + ['AB-001', 'TGN1412']
    + [
        'Adalimumab',
        'Bevacizumab',
        'Pembrolizumab',
        'Trastuzumab',
        'Cetuximab',
        'Omalizumab',
        'Atezolizumab',
        'Ipilimumab',
        'Tremelimumab',
        'Natalizumab',
        'Basiliximab',
        'Ganitumab',
        'Vesencumab',
        'Golimumab',
        'Evolocumab',
    ]  # 15
)


def test_lineage_assignment():
    assert assign_lineage('R1-006') == 'R1'
    assert assign_lineage('AB-001') == 'R1'  # clusters with R1 by identity
    assert assign_lineage('R2-012') == 'R2'
    assert assign_lineage('TGN1412') == 'therapeutic'
    assert assign_lineage('Pembrolizumab') == 'therapeutic'


def test_every_entity_assigned_to_exactly_one_fold():
    folds, _ = lineage_stratified_folds(ENTITIES, n_splits=5)
    assert len(folds) == len(ENTITIES)
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}
    assert (folds >= 0).all()


def test_train_val_partition_is_disjoint_and_complete():
    folds, _ = lineage_stratified_folds(ENTITIES, n_splits=5)
    for fold in range(5):
        train_idx, val_idx = fold_indices(folds, fold)
        assert set(train_idx).isdisjoint(val_idx)
        assert len(train_idx) + len(val_idx) == len(ENTITIES)
        assert len(val_idx) > 0


def test_each_fold_has_proportional_lineage_representation():
    folds, _ = lineage_stratified_folds(ENTITIES, n_splits=5)
    comp = composition(ENTITIES, folds)
    # every fold must contain at least one of each major lineage (proportional, not isolated)
    for row in comp:
        assert row['R1'] >= 1
        assert row['R2'] >= 1
        assert row['therapeutic'] >= 1


def test_deterministic_given_seed():
    a, _ = lineage_stratified_folds(ENTITIES, n_splits=5, seed=42)
    b, _ = lineage_stratified_folds(ENTITIES, n_splits=5, seed=42)
    assert np.array_equal(a, b)
