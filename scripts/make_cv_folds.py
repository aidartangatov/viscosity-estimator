"""Generate the canonical lineage-aware CV fold assignment and write it to disk.

Writes ``datasets/full_dataset/cv_folds.csv`` (Entity, lineage, fold) so that
every model (ResNet-3D, ESM2 baseline, ...) is evaluated on the *same* split.

Run:
  python scripts/make_cv_folds.py
"""
from pathlib import Path

import sys
import pandas as pd
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quannet.cv import composition, lineage_stratified_folds  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_csv', type=Path, default=root / 'datasets' / 'full_dataset' / 'dataset.csv')
    ap.add_argument('--out', type=Path, default=root / 'datasets' / 'full_dataset' / 'cv_folds.csv')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset_csv).drop_duplicates('Entity').reset_index(drop=True)
    entities = df['Entity'].tolist()
    folds, strata = lineage_stratified_folds(entities, n_splits=args.n_splits, seed=args.seed)

    out_df = pd.DataFrame({'Entity': entities, 'lineage': strata, 'fold': folds})
    out_df.to_csv(args.out, index=False)
    print(f'Wrote {len(out_df)} rows -> {args.out}')

    print('\nPer-fold lineage composition:')
    comp = pd.DataFrame(composition(entities, folds))
    print(comp.to_string(index=False))


if __name__ == '__main__':
    main()
