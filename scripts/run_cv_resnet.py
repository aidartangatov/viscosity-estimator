"""Supervised ResNet-3D cross-validation on the lineage-aware folds.

Trains a from-scratch ResNet-3D per fold on the precomputed ESP tensor cache
(default: ``runs/train2``), collects out-of-fold (OOF) predictions, and reports
Spearman / MAE / R2 in the same format as the ESM2 baseline so the two are
directly comparable. This is the "without SSL" baseline for the SSL experiment.

Reuses the package's LitModel / ResNet3DModule / QuanDataset / QuanSampler and
reads the .npy cache directly (no Docker / APBS / check_dataset).

Run (in the `phabnet` env):
  python scripts/run_cv_resnet.py
  python scripts/run_cv_resnet.py --max_epochs 60 --accelerator gpu
"""
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping

import sys
import json
import numpy as np
import torch
import pandas as pd
import argparse
import lightning as L

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quannet.tasks import LitModel  # noqa: E402
from quannet.config import get_config  # noqa: E402
from quannet.dataset import QuanDataset, QuanSampler  # noqa: E402
from quannet.models.resnet3d.model import ResNet3DModule  # noqa: E402


def index_cache(cache_root: Path) -> dict:
    """Map Entity -> sorted list of its augmentation .npy paths."""
    idx = defaultdict(list)
    for npy in Path(cache_root).rglob('*.npy'):
        idx[npy.parent.name].append(npy)
    return {e: sorted(paths) for e, paths in idx.items()}


def build_tensors(entities, target_map, cache_idx, n_aug):
    """Stack augmentations for a list of entities into model-ready tensors.

    Returns (X, y, structure_ids):
      X: (n_entities * n_aug, 1, D, H, W) float tensor
      y: (n_entities * n_aug, 1) float tensor
      structure_ids: (n_entities * n_aug,) int array (augmentations share an id)
    """
    arrays, targets, sids = [], [], []
    for i, ent in enumerate(entities):
        files = cache_idx[ent][:n_aug]
        if len(files) < n_aug:
            raise ValueError(f'{ent}: only {len(files)} augmentations cached, need {n_aug}')
        for f in files:
            arrays.append(np.load(f))
            targets.append(target_map[ent])
            sids.append(i)
    X = np.expand_dims(np.stack(arrays), 1).astype('float32')
    return (
        torch.from_numpy(X),
        torch.tensor(targets, dtype=torch.float32).view(-1, 1),
        np.array(sids),
    )


@torch.no_grad()
def predict_per_entity(model, X, sids, batch_size, device):
    """Mean-pool augmentation predictions back to one value per entity (ordered)."""
    model.eval().to(device)
    preds = []
    for i in range(0, len(X), batch_size):
        out = model(X[i : i + batch_size].to(device))
        preds.append(out.cpu().numpy().ravel())
    preds = np.concatenate(preds)
    # average the augmentations of each structure id
    n_entities = sids.max() + 1
    return np.array([preds[sids == s].mean() for s in range(n_entities)])


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_csv', type=Path, default=root / 'datasets' / 'full_dataset' / 'dataset.csv')
    ap.add_argument('--cv_folds', type=Path, default=root / 'datasets' / 'full_dataset' / 'cv_folds.csv')
    ap.add_argument('--cache_root', type=Path, default=root / 'runs' / 'train2')
    ap.add_argument('--output_dir', type=Path, default=root / 'runs' / 'resnet3d_cv_lineage')
    ap.add_argument('--n_aug', type=int, default=3)
    ap.add_argument('--max_epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--accelerator', type=str, default='auto')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if (args.accelerator in ('auto', 'gpu') and torch.cuda.is_available()) else 'cpu'
    print(f'Device: {device}')

    df = pd.read_csv(args.dataset_csv).drop_duplicates('Entity')
    target_map = dict(zip(df['Entity'], df['Viscosity_at_150'].astype(float)))
    fold_df = pd.read_csv(args.cv_folds)
    cache_idx = index_cache(args.cache_root)

    missing = [e for e in fold_df['Entity'] if e not in cache_idx]
    if missing:
        raise ValueError(f'{len(missing)} entities have no cached ESP tensors: {missing}')

    folds = sorted(fold_df['fold'].unique())
    oof = {}  # entity -> predicted
    fold_val_loss = {}

    for fold in folds:
        L.seed_everything(args.seed, workers=True)
        train_ents = fold_df[fold_df['fold'] != fold]['Entity'].tolist()
        val_ents = fold_df[fold_df['fold'] == fold]['Entity'].tolist()

        Xtr, ytr_raw, sid_tr = build_tensors(train_ents, target_map, cache_idx, args.n_aug)
        Xva, yva_raw, sid_va = build_tensors(val_ents, target_map, cache_idx, args.n_aug)

        # Standardize the target on TRAIN statistics only (no leakage). The raw
        # viscosity range (6-310) made the Huber-trained head fail to learn the
        # output scale; training in z-space fixes magnitude. Predictions are
        # inverted back to raw cP below, so metrics stay on the original scale.
        mu = float(ytr_raw.mean())
        sd = float(ytr_raw.std(unbiased=False)) or 1.0
        ytr = (ytr_raw - mu) / sd
        yva = (yva_raw - mu) / sd

        train_loader = DataLoader(
            QuanDataset(Xtr, ytr),
            batch_size=args.batch_size,
            sampler=QuanSampler(sid_tr, batch_size=args.batch_size),
            num_workers=0,
        )
        val_loader = DataLoader(QuanDataset(Xva, yva), batch_size=args.batch_size, num_workers=0)

        model = ResNet3DModule()
        cfg = get_config(overrides={'lr': args.lr, 'weight_decay': args.weight_decay, 'model_arch': 'resnet3d'})
        lit = LitModel(model=model, args=cfg)
        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            deterministic=True,
        )
        trainer.fit(lit, train_loader, val_loader)
        fold_val_loss[int(fold)] = float(trainer.callback_metrics.get('val_loss', float('nan')))

        val_preds = predict_per_entity(model, Xva, sid_va, args.batch_size, device) * sd + mu
        for ent, p in zip(val_ents, val_preds):
            oof[ent] = float(p)
        print(f'fold {fold}: {len(val_ents)} val entities, val_loss={fold_val_loss[int(fold)]:.3f}')

    entities = fold_df['Entity'].tolist()
    y_true = np.array([target_map[e] for e in entities])
    y_pred = np.array([oof[e] for e in entities])

    metrics = {
        'spearman': float(spearmanr(y_true, y_pred).correlation),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'model_arch': 'resnet3d',
        'cv': 'lineage_folds',
        'n_splits': len(folds),
        'n_samples': len(entities),
        'n_aug': args.n_aug,
        'max_epochs': args.max_epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'target_standardized': True,
        'fold_val_loss': fold_val_loss,
    }
    pd.DataFrame(
        {
            'Entity': entities,
            'lineage': [fold_df.set_index('Entity').loc[e, 'lineage'] for e in entities],
            'fold': [int(fold_df.set_index('Entity').loc[e, 'fold']) for e in entities],
            'Viscosity_at_150': y_true,
            'predicted': y_pred,
        }
    ).to_csv(args.output_dir / 'predictions.csv', index=False)
    with open(args.output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(
        f"\nResNet-3D (lineage CV): Spearman={metrics['spearman']:.4f}  "
        f"MAE={metrics['mae']:.3f}  R2={metrics['r2']:.4f}"
    )
    print(f'Saved to {args.output_dir}')


if __name__ == '__main__':
    main()
