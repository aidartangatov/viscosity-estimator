"""Fine-tune (or linear-probe) an SSL-pretrained ResNet-3D encoder on the
lineage-aware folds, and report metrics directly comparable to
``scripts/run_cv_resnet.py`` (the "without SSL" baseline) and the ESM2
baseline (``quannet.baselines.esm2``).

Same lineage-stratified 5-fold CV, same OOF-aggregation, same
predictions.csv/metrics.json schema as ``run_cv_resnet.py`` - only the model
construction differs: the encoder trunk is initialized from an SSL
pretraining checkpoint (``quannet.ssl.pretrain``'s ``encoder.pt``), and can
optionally stay frozen (linear probing) instead of being fully fine-tuned.

Run (in the `phabnet` env):
  python scripts/finetune_ssl_resnet.py --ssl_checkpoint runs/ssl_vicreg/encoder.pt
  python scripts/finetune_ssl_resnet.py --ssl_checkpoint runs/ssl_vicreg/encoder.pt --freeze_encoder
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
import torch.nn as nn
import lightning as L

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quannet.tasks import LitModel  # noqa: E402
from quannet.config import get_config  # noqa: E402
from quannet.dataset import QuanDataset, QuanSampler  # noqa: E402
from quannet.ssl.clearml_utils import (  # noqa: E402
    init_task,
    add_clearml_args,
    resolve_data_dirs,
    ClearMLMetricsLogger,
    resolve_ssl_checkpoint,
    resolve_ssl_architecture,
)
from quannet.models.resnet3d.model import ResNet3DModule  # noqa: E402


class FrozenEncoderModel(nn.Module):
    """Wraps a ResNet3DModule, optionally freezing its trunk (conv/bn/blocks).

    Freezing sets requires_grad=False on the trunk parameters AND keeps the
    trunk's submodules in eval() mode even while the rest of the model
    trains, so BatchNorm running stats and the ResBlocks' internal dropout
    don't drift away from what pretraining learned - this is what makes it a
    linear probe rather than "fine-tuning starting from a good init".
    """

    def __init__(self, backbone: ResNet3DModule, freeze: bool):
        super().__init__()
        self.backbone = backbone
        self.freeze = freeze
        if freeze:
            for module in (backbone.conv, backbone.bn, backbone.blocks):
                for p in module.parameters():
                    p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.conv.eval()
            self.backbone.bn.eval()
            self.backbone.blocks.eval()
        return self

    def forward(self, x):
        return self.backbone(x)


def index_cache(cache_root: Path) -> dict:
    """Map Entity -> sorted list of its augmentation .npy paths."""
    idx = defaultdict(list)
    for npy in Path(cache_root).rglob('*.npy'):
        idx[npy.parent.name].append(npy)
    return {e: sorted(paths) for e, paths in idx.items()}


def build_tensors(entities, target_map, cache_idx, n_aug):
    """Stack augmentations for a list of entities into model-ready tensors."""
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
    n_entities = sids.max() + 1
    return np.array([preds[sids == s].mean() for s in range(n_entities)])


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_csv', type=Path, default=root / 'datasets' / 'full_dataset' / 'dataset.csv')
    ap.add_argument('--cv_folds', type=Path, default=root / 'datasets' / 'full_dataset' / 'cv_folds.csv')
    ap.add_argument('--cache_root', type=Path, default=root / 'runs' / 'train2_fixed' / 'artefacts')
    ap.add_argument(
        '--ssl_checkpoint',
        type=Path,
        default=None,
        help='encoder.pt from quannet.ssl.pretrain; omit for a from-scratch encoder (same as run_cv_resnet.py)',
    )
    ap.add_argument(
        '--clearml-ssl-checkpoint',
        default=None,
        metavar='TASK_ID',
        help='pull the encoder from this ClearML pretrain Task\'s output model instead of --ssl_checkpoint',
    )
    ap.add_argument(
        '--freeze_encoder',
        action='store_true',
        help='linear-probe: freeze the encoder trunk, only train the regression head',
    )
    ap.add_argument(
        '--n-channels',
        type=int,
        nargs='+',
        default=[4, 8, 16, 32],
        help='ResNet3DModule channel widths per stage - MUST match the checkpoint being loaded. '
        'Auto-overridden when --clearml-ssl-checkpoint carries an encoder_arch artifact.',
    )
    ap.add_argument(
        '--n-blocks',
        type=int,
        nargs='+',
        default=[2, 2, 2, 2],
        help='ResNet3DModule residual-block count per stage - MUST match the checkpoint being loaded.',
    )
    ap.add_argument('--output_dir', type=Path, default=root / 'runs' / 'resnet3d_ssl_cv_lineage')
    ap.add_argument('--n_aug', type=int, default=3)
    ap.add_argument('--max_epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--accelerator', type=str, default='auto')
    ap.add_argument('--seed', type=int, default=42)
    add_clearml_args(ap)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task = init_task(args.clearml_project, args.clearml_task_name or args.output_dir.name, args.clearml_tags)
    if task is not None:
        task.connect(vars(args))
    if args.clearml_dataset:
        args.cache_root = Path(resolve_data_dirs([], args.clearml_dataset)[0])
    args.ssl_checkpoint = resolve_ssl_checkpoint(args.ssl_checkpoint, args.clearml_ssl_checkpoint)
    arch = resolve_ssl_architecture(args.clearml_ssl_checkpoint)
    if arch is not None:
        args.n_channels, args.n_blocks = arch['n_channels'], arch['n_blocks']
        print(f"Encoder architecture from ClearML task: n_channels={args.n_channels} n_blocks={args.n_blocks}")

    device = 'cuda' if (args.accelerator in ('auto', 'gpu') and torch.cuda.is_available()) else 'cpu'
    print(f'Device: {device}')
    if args.ssl_checkpoint:
        print(f'SSL checkpoint: {args.ssl_checkpoint} (freeze_encoder={args.freeze_encoder})')
    else:
        print('No SSL checkpoint given - training the encoder from scratch (same as run_cv_resnet.py)')

    df = pd.read_csv(args.dataset_csv).drop_duplicates('Entity')
    target_map = dict(zip(df['Entity'], df['Viscosity_at_150'].astype(float)))
    fold_df = pd.read_csv(args.cv_folds)
    cache_idx = index_cache(args.cache_root)

    missing = [e for e in fold_df['Entity'] if e not in cache_idx]
    if missing:
        print(f'WARNING: dropping {len(missing)} entities with no cached ESP tensors: {missing}')
        fold_df = fold_df[~fold_df['Entity'].isin(missing)].reset_index(drop=True)

    folds = sorted(fold_df['fold'].unique())
    oof = {}
    fold_val_loss = {}

    for fold in folds:
        L.seed_everything(args.seed, workers=True)
        train_ents = fold_df[fold_df['fold'] != fold]['Entity'].tolist()
        val_ents = fold_df[fold_df['fold'] == fold]['Entity'].tolist()

        Xtr, ytr_raw, sid_tr = build_tensors(train_ents, target_map, cache_idx, args.n_aug)
        Xva, yva_raw, sid_va = build_tensors(val_ents, target_map, cache_idx, args.n_aug)

        # Standardize the target on TRAIN statistics only (no leakage) - see
        # run_cv_resnet.py for why (raw viscosity range made Huber loss fail
        # to learn the output scale).
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

        backbone = ResNet3DModule(n_channels=args.n_channels, n_blocks=args.n_blocks)
        if args.ssl_checkpoint:
            state_dict = torch.load(args.ssl_checkpoint, map_location='cpu')
            backbone.load_state_dict(state_dict, strict=True)
        model = FrozenEncoderModel(backbone, freeze=args.freeze_encoder)

        cfg = get_config(overrides={'lr': args.lr, 'weight_decay': args.weight_decay, 'model_arch': 'resnet3d'})
        lit = LitModel(model=model, args=cfg)
        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=args.patience, mode='min'),
                ClearMLMetricsLogger(task, series=f'fold_{fold}'),
            ],
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
        'n_channels': args.n_channels,
        'n_blocks': args.n_blocks,
        'ssl_checkpoint': str(args.ssl_checkpoint) if args.ssl_checkpoint else None,
        'freeze_encoder': bool(args.freeze_encoder),
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
    predictions_path = args.output_dir / 'predictions.csv'
    metrics_path = args.output_dir / 'metrics.json'
    pd.DataFrame(
        {
            'Entity': entities,
            'lineage': [fold_df.set_index('Entity').loc[e, 'lineage'] for e in entities],
            'fold': [int(fold_df.set_index('Entity').loc[e, 'fold']) for e in entities],
            'Viscosity_at_150': y_true,
            'predicted': y_pred,
        }
    ).to_csv(predictions_path, index=False)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(
        f"\nResNet-3D SSL fine-tune (lineage CV): Spearman={metrics['spearman']:.4f}  "
        f"MAE={metrics['mae']:.3f}  R2={metrics['r2']:.4f}"
    )
    print(f'Saved to {args.output_dir}')

    if task is not None:
        logger = task.get_logger()
        for key in ('spearman', 'mae', 'r2'):
            logger.report_single_value(name=key, value=metrics[key])
        task.upload_artifact(name='predictions', artifact_object=str(predictions_path))
        task.upload_artifact(name='metrics', artifact_object=str(metrics_path))
        task.close()


if __name__ == '__main__':
    main()
