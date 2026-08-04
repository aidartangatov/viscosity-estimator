"""
One-shot: rewrite datasets/full_dataset/dataset.csv to add `path` and `target`
columns required by the CNN3D trainer's `check_dataset` (see
src/quannet/trainer.py). The trainer reads only `path,target`; the ESM2 baseline
keeps reading `Entity,Viscosity_at_150` from the same file.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'datasets' / 'full_dataset'
CSV = DATA_DIR / 'dataset.csv'


def main():
    df = pd.read_csv(CSV)
    df = df.drop_duplicates(subset='Entity', keep='first').reset_index(drop=True)

    df['path'] = df['Entity'].apply(lambda e: f'./full_dataset/{e}.pdb')
    df['target'] = df['Viscosity_at_150']
    df['_exists'] = df['path'].apply(lambda p: (DATA_DIR / p).exists())

    missing = df[~df['_exists']]
    print(f'Missing PDB for {len(missing)} entries: {missing["Entity"].tolist()}')

    df = df[df['_exists']].drop(columns='_exists').reset_index(drop=True)
    df = df[['path', 'target', 'Entity', 'Viscosity_at_150']]
    df.to_csv(CSV, index=False)
    print(f'Wrote {len(df)} rows to {CSV}')
    print(df.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
