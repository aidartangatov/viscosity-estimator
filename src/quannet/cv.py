"""Lineage-aware, proportional cross-validation splits for the antibody viscosity dataset.

Empirical lineage structure (measured by pairwise sequence identity over the 56
labeled antibodies; see ``scripts/analyze_lineages.py``):

  * R1 (17), R2 (22) and AB-001 form ONE near-identical mutational family
    (within-identity ~0.99, cross R1<->R2 ~0.965). They differ by point
    mutations only, yet span the full viscosity range (R2 ~29-66, R1 ~94-310).
    Holding the whole family out as a single disjoint group yields a degenerate
    split (a test fold with no high-viscosity examples), so the family is NOT
    treated as one opaque group. Instead each series is a *stratum* distributed
    proportionally across folds.
  * The 15 marketed therapeutic mAbs (+ TGN1412) are mutually diverse
    (identity ~0.18-0.83) and all low-viscosity; each is essentially its own
    lineage.

Strata used for the proportional StratifiedKFold:

  ===============  =======================  ================
  stratum          members                  viscosity level
  ===============  =======================  ================
  ``R1``           R1-* and AB-001          high
  ``R2``           R2-*                     medium
  ``therapeutic``  marketed mAbs, TGN1412   low
  ===============  =======================  ================

NOTE: because R1/R2 variants are point mutants, this split measures *within-
lineage* generalization (how mutations shift viscosity), not novel-scaffold
generalization. That is the scientifically meaningful target for this dataset;
report it as such when comparing against the SSL-pretrained model.
"""
from typing import List, Tuple, Sequence
from sklearn.model_selection import StratifiedKFold

import numpy as np

LINEAGES = ('R1', 'R2', 'therapeutic')


def assign_lineage(entity: str) -> str:
    """Map an Entity id to its lineage stratum.

    Validated against sequence-identity clustering: AB-001 belongs with R1
    (0.989 identity); every marketed mAb and TGN1412 is grouped under
    ``therapeutic``.
    """
    if entity.startswith('R1') or entity == 'AB-001':
        return 'R1'
    if entity.startswith('R2'):
        return 'R2'
    return 'therapeutic'


def lineage_stratified_folds(
    entities: Sequence[str], n_splits: int = 5, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each entity to one of ``n_splits`` folds, proportional per lineage.

    Args:
        entities: ordered Entity ids.
        n_splits: number of CV folds.
        seed: RNG seed for reproducible fold assignment.

    Returns:
        (folds, strata): ``folds[i]`` is the fold index (0..n_splits-1) of
        ``entities[i]``; ``strata[i]`` is its lineage label.
    """
    strata = np.array([assign_lineage(e) for e in entities])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(entities), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(entities)), strata)):
        folds[val_idx] = fold
    return folds, strata


def fold_indices(folds: np.ndarray, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) for a given fold from a fold-assignment array."""
    val_idx = np.where(folds == fold)[0]
    train_idx = np.where(folds != fold)[0]
    return train_idx, val_idx


def composition(entities: Sequence[str], folds: np.ndarray) -> List[dict]:
    """Per-fold lineage counts, for sanity-checking proportionality."""
    strata = np.array([assign_lineage(e) for e in entities])
    rows = []
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        row = {'fold': int(fold), 'n': int(mask.sum())}
        for lin in LINEAGES:
            row[lin] = int(((strata == lin) & mask).sum())
        rows.append(row)
    return rows
