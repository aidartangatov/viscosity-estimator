"""Unified Docker ENTRYPOINT for the quannet-ssl image: dispatches to
quannet.ssl.pretrain or finetune_ssl_resnet based on the first CLI argument,
so one image/entrypoint covers both stages instead of needing a different
`docker run --entrypoint` override per command.

    docker run --gpus all ... quannet-ssl pretrain --method vicreg \\
        --data-dirs /data/sabdab/artefacts --output-dir /app/runs/ssl_vicreg --accelerator gpu
    docker run --gpus all ... quannet-ssl finetune --ssl_checkpoint runs/ssl_vicreg/encoder.pt
    docker run --gpus all ... quannet-ssl finetune --ssl_checkpoint runs/ssl_vicreg/encoder.pt --freeze_encoder

Every flag after the subcommand is passed straight through to that script's
own argparse - see `python -m quannet.ssl.pretrain --help` / the docstring at
the top of scripts/finetune_ssl_resnet.py for the full list, including the
ClearML flags (--clearml-project, --clearml-dataset, ...).
"""
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = ('pretrain', 'finetune')


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        prog = Path(sys.argv[0]).name
        print(f'Usage: {prog} {{{"|".join(COMMANDS)}}} [args...]', file=sys.stderr)
        sys.exit(1)

    command, rest = sys.argv[1], sys.argv[2:]
    sys.argv = [f'{sys.argv[0]} {command}', *rest]

    if command == 'pretrain':
        from quannet.ssl.pretrain import main as run
    else:
        from finetune_ssl_resnet import main as run
    run()


if __name__ == '__main__':
    main()
