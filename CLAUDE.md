# ViscosityEstimator (`quannet`) — Claude Notes

ResNet-3D viscosity predictor for antibody solutions. PyTorch Lightning trainer,
APBS-based input pipeline (dockerized), single CLI entry point (`quannet`).

The user-facing usage doc is `README.md`. This file is a code map and gotchas
list for engineering work.

## Directory layout (relevant pieces)

```
src/quannet/
├── __init__.py              # exposes `QuanNet` Python API
├── config/
│   ├── __init__.py          # `entrypoint()` — the `quannet` CLI dispatcher
│   └── default.yaml         # canonical hyperparameter defaults
├── dataset.py               # `QuanDataset`, `QuanSampler` (groups augmentations)
├── make_inputs.py           # PDB → PQR → APBS → ESP tensor pipeline (~700 lines)
├── preprocessor.py          # orchestrates make_inputs vs. precomputed-input mode
├── predictor.py             # `Predictor` — inference interface
├── trainer.py               # `Trainer` — Lightning training orchestrator
├── model.py                 # `QuanNet` high-level Python API wrapper
├── tasks/                   # `BaseModel` and task-specific heads
├── models/
│   ├── __init__.py          # exports `CNN3D` only — `from .cnn3d import CNN3D`
│   ├── cnn3d/model.py       # **`CNN3DModule`** — the network the package actually uses
│   └── resnet3d/model.py    # `ResNet3DModule` — defined but unreferenced (see KNOWN DRIFT)
├── docker/
│   └── make_inputs.py       # runs APBS+pdb2pqr inside the `quannet:latest` container
└── utils/                   # misc helpers
Dockerfile                   # builds `quannet:latest` (APBS 3.4.1 + pdb2pqr)
setup.py                     # installs `quannet` console script
```

## CLI entry point

`setup.py` registers `quannet = quannet.config:entrypoint`. The dispatcher in
`src/quannet/config/__init__.py:entrypoint()` parses `key=value` overrides
against `default.yaml` and routes to `train` / `predict`.

```bash
quannet train   dataset=datasets/quannet_test
quannet predict model=models/quannet.pt structures=datasets/quannet_test/structures
```

Python API equivalent (see `__init__.py`):

```python
from quannet import QuanNet
QuanNet("models/quannet.pt").predict(structures="…/structures")
```

## Pipeline

1. **Inputs** (`make_inputs.py`):
   - `euler_rotate()` rotates atom coordinates by random Euler angles.
   - `get_esp_array()` runs `pdb2pqr → APBS` for one (rotated) PDB and returns
     a 3-D numpy array. APBS is invoked **inside the `quannet:latest` Docker
     container** via `src/quannet/docker/make_inputs.py`.
   - `get_esp_array_rotations()` parallelizes `num_augmentations` rotated
     copies per structure (multiprocessing pool of size `processes`).
   - Water-inaccessible voxels are zeroed using a shell of width
     `shell_width` (default 2.0 Å).
2. **Dataset** (`dataset.py`): wraps the saved `.npy` ESP arrays with a custom
   `QuanSampler` that keeps all augmentations of one structure inside the same
   batch (important — see Gotchas).
3. **Model** (`models/cnn3d/model.py`): `CNN3DModule` — the model wired into
   the package as `quannet.QuanNet` via `from quannet.models import CNN3D as
   QuanNet` in `src/quannet/__init__.py`. **No residual connections**, no BN.
   Architecture (with `grid_dim=96`, the first 1×1×1 block is skipped):
   `Conv3d(1, 2) + ReLU + MaxPool` → … 5× channel-doubling `Conv3d + ReLU +
   MaxPool` blocks (2→4→8→16→32→64) → `Conv3d(64, 1024)` + ReLU (no final
   pool) → `Flatten` → `Dropout(0.05)` → `Linear(1024, 1) + ReLU`. About
   1.8 M parameters. **The `ResNet3DModule` in `models/resnet3d/` exists but
   is unreferenced**; switching to it would require editing
   `src/quannet/models/__init__.py`.
4. **Training** (`trainer.py`): Lightning `Trainer`, stratified train/val
   split via `sklearn`, Adam optimizer, `lr` and `weight_decay` from
   `default.yaml`.

## Configuration

`config/default.yaml` is the single source of truth for hyperparameters. Key
fields:

| Field               | Default | Notes |
|---------------------|---------|-------|
| `grid_dim`          | 96      | ESP grid is 96³ voxels |
| `grid_spacing`      | 0.75    | Å between grid points |
| `shell_width`       | 2.0     | Å — water-accessible shell thickness |
| `num_augmentations` | 5       | rotated PDB copies per structure |
| `lr`                | 0.0001  | Adam learning rate |
| `weight_decay`      | 1e-5    | L2 regularization |
| `batch_size`        | 4       |       |
| `max_epochs`        | 3       | dummy default; override for real runs |
| `val_size`          | 0.2     | stratified split fraction |
| `processes`         | 4       | APBS preprocessing workers |
| `accelerator`       | auto    | Lightning device selection |

Override at the CLI: `quannet train batch_size=8 max_epochs=200 lr=5e-5 …`.
Or pass `config=path/to/custom.yaml` to load a full overlay.

## ⚠️ KNOWN DRIFT — paper draft vs. code

The paper draft and the shipped `default.yaml` disagree on training
hyperparameters. Reconcile before submission.

| Quantity            | Paper claim                          | Code reality |
|---------------------|--------------------------------------|----------------|
| Architecture        | ResNet with 4 blocks × 4 Conv3d layers + skip connections, BN, Dropout 10% | `CNN3DModule`: plain 7-layer Conv3d + MaxPool stack, **no residuals, no BN**, Dropout 5%. `ResNet3DModule` exists in source but is never imported. |
| Learning rate       | 1e-6                                 | 1e-4 (`default.yaml`) |
| Weight decay (L2)   | 0.001                                | 1e-5 (`default.yaml`) |
| Batch size          | 2                                    | 4 (`default.yaml`) |
| Activation on output | (regression, none implied)          | `nn.ReLU()` on the regression head — any negative pre-activation clamps to 0, which **caused all-zero predictions** when running the bundled `models/quannet.pt` against `datasets/quannet_test/structures` |
| Test set size       | "16 structures"                      | `quannet_test/` ships 7 PDBs; `datasets/quannet_test/` ships 5 |

Either the paper text needs updating to match the configuration that produced
the reported 89% / 86% AUC numbers, or a separate `paper.yaml` snapshot of the
hyperparameters that produced those numbers should be checked in alongside
the trained weights.

## Augmentation: current vs. planned

- **Current**: random Euler rotation of PDB atom coordinates, then re-run
  APBS. High chemical fidelity but expensive (one APBS call per augmentation).
- **Planned (hypothesis to test)**: rotate the **ESP tensor** directly with
  a 3-D interpolating rotation. Cheaper (no APBS recompute), and may give
  finer angular sampling. Risks: voxelization artifacts and loss of the
  water-accessibility shell semantics. If implemented, gate it behind a
  config flag (e.g. `augmentation_space: pdb | tensor`) so the original
  pipeline remains reproducible.

## Gotchas

- `make_inputs.py` shells out to APBS via Docker. The Mac host needs a running
  Docker daemon and the `quannet:latest` image (`docker build -t quannet .`).
- `QuanSampler` keeps a structure's augmentations together in a batch. Don't
  swap in a vanilla shuffler without thinking about the train/val leakage that
  random shuffling would introduce.
- `default.yaml` defaults to `mode: predict` and `max_epochs: 3`. Real
  training runs **must** override `max_epochs`.
- `precomputed_input: true` skips the Docker/APBS step and reads `.npy` ESP
  arrays directly. Use this when iterating on the model only.
- No automated test suite exists. There are manual notebooks under
  `../notebooks/test_*.ipynb` but no `pytest`. New work should add at least
  smoke tests for `make_inputs`, `QuanDataset`, and `CNN3DModule.forward`.

## Known runtime gotchas (observed during end-to-end test, 2026-05-12)

- **PyTorch 2.0.1 has no Conv3d on MPS.** Training on Apple Silicon needs
  `accelerator=cpu` explicitly, e.g.
  `quannet train dataset=… accelerator=cpu`. Bumping `torch` to ≥ 2.1 added
  MPS Conv3d support and would let the trainer use the M-series GPU.
- **Bundled `models/quannet.pt` predicts 0.0 for every input** on
  `datasets/quannet_test/structures`. Pipeline mechanics are fine
  (16/16 weights load, APBS runs, ESP tensors build) — the issue is either an
  undertrained checkpoint or the `nn.ReLU()` head clamping all outputs to 0.
  Don't trust quantitative outputs from the bundled model without retraining.
- **Cross-platform Docker build verified.** The `--platform=linux/amd64` pin
  in the Dockerfile works on Apple Silicon under emulation; APBS runs inside
  the container with the expected `"ERROR -- APBS input file not specified!"`
  smoke response when invoked with no arguments.

## Install reality (Python 3.11.15, Apple Silicon)

The requirements files were originally pinned to versions that don't all
install cleanly on Python 3.11 with modern pip / setuptools. After the
end-to-end test pass the pinned set is:

- `inputs.txt`: `numba==0.57.1` (was malformed `numba=0.57.1`),
  `docker==7.1.0` (was `docker==6.1.3`, which crashes against `urllib3 ≥ 2`
  with `Not supported URL scheme http+docker`).
- `model.txt`: `pandas==2.0.3` (was `1.3.3`, no Py 3.11 wheel),
  added `lightning-cloud<0.6` (Lightning 2.0.7 expects the old
  `AppinstancesIdBody` API), added `setuptools<70` (Lightning 2.0.7 still
  calls `pkg_resources.declare_namespace`).
- `setup.py`: dropped the `pkg_resources` import (modern PEP 517 isolated
  build envs ship without it), added `package_dir={'': 'src'}` and
  `packages=find_packages(where='src')` so the package is actually packaged,
  and `package_data={'quannet': ['config/*.yaml']}` so `default.yaml` ships.
- `MANIFEST.in`: `recursive-include src/quannet *.yaml` (was the malformed
  `recursive-include *.yaml` with no directory, so `default.yaml` never
  shipped and first import crashed on missing config).

If a future bump targets a current PyTorch/Lightning stack, dropping all of
the above pins simultaneously is the cleanest path — they are all
workarounds for the original 2023-era pin set.
