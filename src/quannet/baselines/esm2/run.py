"""
ESM2 baseline for antibody viscosity regression.

Pipeline:
  1. Read entity → viscosity targets from a CSV (Entity, Viscosity_at_150).
  2. For each entity, locate the PDB file in `structures_dir`, extract H+L
     sequences via Biopython.
  3. Run frozen ESM2 on every chain once, mean-pool over residue positions
     (excluding CLS/EOS) and cache as a {entity: 2D-tensor} dict.
  4. Concatenate H and L embeddings into a single feature vector per entity.
  5. Cross-validate RidgeCV (5-fold) on (X, y); report Spearman, MAE, R².

Run:
  python -m quannet.baselines.esm2.run \\
      --dataset_csv datasets/full_dataset/dataset.csv \\
      --structures_dir datasets/full_dataset/full_dataset \\
      --output_dir runs/esm2_baseline
"""
from typing import Dict, List, Tuple
from pathlib import Path
from quannet.utils import LOGGER
from quannet.baselines.esm2.embed import DEFAULT_MODEL, embed_sequences, load_embeddings, save_embeddings
from quannet.baselines.esm2.train import cross_validate_ridge
from quannet.baselines.esm2.sequences import HEAVY_LIGHT, extract_chain_sequences

import json
import numpy as np
import torch
import pandas as pd
import argparse


def collect_sequences(
    entities: List[str],
    structures_dir: Path,
) -> Tuple[Dict[str, str], List[str]]:
    """Return {f"{entity}::H": seq, f"{entity}::L": seq} and the list of usable entities."""
    chain_seqs: Dict[str, str] = {}
    usable: List[str] = []
    missing_pdb: List[str] = []
    missing_chain: List[str] = []

    for entity in entities:
        pdb_path = structures_dir / f'{entity}.pdb'
        if not pdb_path.exists():
            missing_pdb.append(entity)
            continue
        seqs = extract_chain_sequences(pdb_path)
        if not all(c in seqs and seqs[c] for c in HEAVY_LIGHT):
            missing_chain.append(entity)
            continue
        for chain_id in HEAVY_LIGHT:
            chain_seqs[f'{entity}::{chain_id}'] = seqs[chain_id]
        usable.append(entity)

    if missing_pdb:
        LOGGER.warning(f'No PDB for {len(missing_pdb)} entities: {missing_pdb}')
    if missing_chain:
        LOGGER.warning(f'Missing H/L chain in {len(missing_chain)} entities: {missing_chain}')
    LOGGER.info(f'Usable entities: {len(usable)}/{len(entities)}')
    return chain_seqs, usable


def build_feature_matrix(
    entities: List[str],
    embeddings: Dict[str, torch.Tensor],
) -> np.ndarray:
    rows = []
    for entity in entities:
        heavy = embeddings[f'{entity}::H']
        light = embeddings[f'{entity}::L']
        rows.append(torch.cat([heavy, light], dim=0).numpy())
    return np.stack(rows, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_csv', type=Path, required=True)
    parser.add_argument('--structures_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument(
        '--cv_folds',
        type=Path,
        default=None,
        help='CSV with columns Entity,fold (e.g. datasets/full_dataset/cv_folds.csv). '
        'When given, defines the CV splits exactly instead of a random KFold.',
    )
    parser.add_argument(
        '--device', type=str, default=None, help='cuda / cpu / mps. Default: cuda if available else cpu.'
    )
    parser.add_argument(
        '--force_reembed', action='store_true', help='Recompute embeddings even if the cache file exists.'
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output_dir / 'embeddings.pt'

    df = pd.read_csv(args.dataset_csv)
    df = df.drop_duplicates(subset='Entity', keep='first').reset_index(drop=True)
    LOGGER.info(f'CSV rows after de-dup: {len(df)}')

    chain_seqs, usable = collect_sequences(df['Entity'].tolist(), args.structures_dir)
    df_usable = df[df['Entity'].isin(usable)].reset_index(drop=True)

    if embeddings_path.exists() and not args.force_reembed:
        LOGGER.info(f'Loading cached embeddings from {embeddings_path}')
        embeddings = load_embeddings(embeddings_path)
        missing = [k for k in chain_seqs if k not in embeddings]
        if missing:
            LOGGER.info(f'Embedding {len(missing)} new chains not in cache ...')
            new_emb = embed_sequences({k: chain_seqs[k] for k in missing}, model_name=args.model, device=args.device)
            embeddings.update(new_emb)
            save_embeddings(embeddings, embeddings_path)
    else:
        embeddings = embed_sequences(chain_seqs, model_name=args.model, device=args.device)
        save_embeddings(embeddings, embeddings_path)
        LOGGER.info(f'Embeddings cached to {embeddings_path}')

    X = build_feature_matrix(df_usable['Entity'].tolist(), embeddings)
    y = df_usable['Viscosity_at_150'].to_numpy(dtype=float)
    LOGGER.info(f'X shape: {X.shape}, y shape: {y.shape}')

    folds = None
    if args.cv_folds is not None:
        fold_df = pd.read_csv(args.cv_folds)
        entity2fold = dict(zip(fold_df['Entity'], fold_df['fold']))
        missing = [e for e in df_usable['Entity'] if e not in entity2fold]
        if missing:
            raise ValueError(f'{len(missing)} usable entities missing from {args.cv_folds}: {missing}')
        folds = np.array([entity2fold[e] for e in df_usable['Entity']])
        LOGGER.info(f'Using lineage-aware folds from {args.cv_folds} ({len(np.unique(folds))} folds)')

    results = cross_validate_ridge(X, y, n_splits=args.n_splits, random_state=args.random_state, folds=folds)
    LOGGER.info(f"Spearman={results['spearman']:.4f}  MAE={results['mae']:.3f}  R²={results['r2']:.4f}")
    LOGGER.info(f"Per-fold alpha: {results['fold_alphas']}")

    preds_df = pd.DataFrame(
        {
            'Entity': df_usable['Entity'],
            'Viscosity_at_150': y,
            'predicted': results['predictions'],
        }
    )
    preds_df.to_csv(args.output_dir / 'predictions.csv', index=False)

    metrics = {k: v for k, v in results.items() if k != 'predictions'}
    metrics['model'] = args.model
    metrics['n_samples'] = int(len(y))
    metrics['n_features'] = int(X.shape[1])
    metrics['n_splits'] = int(len(np.unique(folds))) if folds is not None else args.n_splits
    metrics['cv'] = 'lineage_folds' if folds is not None else 'random_kfold'
    with open(args.output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f'Metrics + predictions saved to {args.output_dir}')


if __name__ == '__main__':
    main()
