"""CLI entry point for SSL pretraining of the ResNet-3D ESP encoder.

Usage:
    python -m quannet.ssl.pretrain --method vicreg --data-dirs <dir1> <dir2> ... --output-dir runs/ssl_vicreg
    python -m quannet.ssl.pretrain --method mae --data-dirs <dir1> --output-dir runs/ssl_mae --max-epochs 20

    # Resume an interrupted run - Lightning checkpoints (periodic + last) are
    # written under <output-dir>/checkpoints/ automatically:
    python -m quannet.ssl.pretrain --method vicreg --data-dirs <dir1> --output-dir runs/ssl_vicreg \\
        --resume-from-checkpoint runs/ssl_vicreg/checkpoints/last.ckpt

Saves the encoder's state_dict (not the pretext-task head) to
``<output-dir>/encoder.pt``, loadable via:
    from quannet.models.resnet3d.model import ResNet3DModule
    encoder = ResNet3DModule()
    encoder.load_state_dict(torch.load('runs/ssl_vicreg/encoder.pt'), strict=True)
"""
from pathlib import Path
from quannet.ssl.mae import MAELitModule
from torch.utils.data import DataLoader, random_split
from quannet.ssl.vicreg import VICRegLitModule
from quannet.ssl.dataset import MAEPretrainDataset, VICRegPretrainDataset
from quannet.ssl.clearml_utils import (
    init_task,
    add_clearml_args,
    resolve_data_dirs,
    upload_output_model,
    ClearMLMetricsLogger,
    ClearMLCheckpointUpload,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from quannet.models.resnet3d.model import ResNet3DModule

import torch
import argparse
import lightning as L


def build_module_and_dataset(method: str, data_dirs, **kwargs):
    encoder = ResNet3DModule()
    if method == 'vicreg':
        dataset = VICRegPretrainDataset(data_dirs)
        module = VICRegLitModule(encoder, **kwargs)
    elif method == 'mae':
        dataset = MAEPretrainDataset(data_dirs)
        module = MAELitModule(encoder, **kwargs)
    else:
        raise ValueError(f'Unknown method: {method!r} (expected "vicreg" or "mae")')
    return encoder, module, dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', choices=['vicreg', 'mae'], required=True)
    ap.add_argument(
        '--data-dirs',
        nargs='+',
        default=[],
        help='one or more ESP artefact-cache roots (in addition to any --clearml-dataset)',
    )
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--max-epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--accelerator', default='auto')
    ap.add_argument('--devices', default='auto')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit-structures', type=int, default=None, help='debug: cap dataset size')
    ap.add_argument(
        '--val-fraction',
        type=float,
        default=0.0,
        help='held-out fraction for a val_loss curve (0 = train-only, matching the original smoke-test path)',
    )
    ap.add_argument(
        '--checkpoint-every-n-epochs',
        type=int,
        default=1,
        help='how often to write a resumable Lightning checkpoint under <output-dir>/checkpoints/',
    )
    ap.add_argument(
        '--resume-from-checkpoint',
        type=Path,
        default=None,
        help='a .ckpt written by a previous run of this script (e.g. checkpoints/last.ckpt)',
    )
    add_clearml_args(ap)
    args = ap.parse_args()

    L.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task = init_task(args.clearml_project, args.clearml_task_name or args.output_dir.name, args.clearml_tags)
    if task is not None:
        task.connect(vars(args))
    args.data_dirs = resolve_data_dirs(args.data_dirs, args.clearml_dataset)
    if not args.data_dirs:
        ap.error('no data: pass --data-dirs and/or --clearml-dataset')

    encoder, module, dataset = build_module_and_dataset(
        args.method,
        args.data_dirs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.limit_structures:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit_structures, len(dataset))))
    print(f'{args.method}: {len(dataset)} training samples from {args.data_dirs}')

    val_loader = None
    if args.val_fraction > 0:
        n_val = max(1, int(len(dataset) * args.val_fraction))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
        )
        print(f'  train/val split: {n_train}/{n_val}')
    else:
        train_dataset = dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    checkpoint_dir = args.output_dir / 'checkpoints'
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_last=True,
        save_top_k=1,
        monitor='val_loss' if val_loader is not None else None,
    )
    clearml_upload_callback = ClearMLCheckpointUpload(task, checkpoint_dir, args.checkpoint_every_n_epochs)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=str(args.output_dir),
        callbacks=[checkpoint_callback, clearml_upload_callback, ClearMLMetricsLogger(task)],
    )
    trainer.fit(
        module,
        train_loader,
        val_loader,
        ckpt_path=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
    )

    encoder_path = args.output_dir / 'encoder.pt'
    torch.save(encoder.state_dict(), encoder_path)
    print(f'Saved pretrained encoder to {encoder_path}')

    upload_output_model(task, encoder_path, name=f'{args.method}_encoder')
    if task is not None:
        task.close()


if __name__ == '__main__':
    main()
