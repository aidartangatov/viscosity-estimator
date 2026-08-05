# Full pipeline: APBS preprocessing -> SSL pretraining -> fine-tuning

Two independent stages, each with its own Docker image, because they need
opposite hardware: preprocessing is CPU-only (APBS has no GPU path);
training needs a GPU to be practical at any real scale.

| Stage | Image | Hardware | Script |
|---|---|---|---|
| A. Data preprocessing | `quannet:latest` (`Dockerfile`, repo root) | CPU | `scripts/extract_antibody_chains.py`, `scripts/build_esp_dataset.py` |
| B. SSL pretrain + fine-tune | `quannet-ssl` (`docker/ssl/Dockerfile`) | GPU | `scripts/ssl_entrypoint.py` (dispatches to `quannet.ssl.pretrain` / `scripts/finetune_ssl_resnet.py`) |

Self-supervised pretraining of the ResNet-3D ESP encoder (`quannet.models.resnet3d.model.ResNet3DModule`)
on unlabeled electrostatic-potential (ESP) tensors, followed by fine-tuning
on the 56-antibody labeled viscosity dataset. Two pretext tasks:

- **v1 - VICReg** (`quannet.ssl.vicreg`): invariance/variance/covariance
  regularization between two rotation-augmented views of the same antibody.
  No target network or negative sampling needed.
- **v2 - MAE** (`quannet.ssl.mae`): mask random cubic blocks of the ESP grid,
  reconstruct them from the unmasked context.

## Quick start on a rented GPU server

Data (ClearML, see B3/B4) and code (GitHub) both already live off-box, so a
freshly rented GPU instance needs nothing pre-staged - just these steps, in
order:

1. **Verify the GPU is actually usable in a container** before building
   anything - if this fails, it's a driver/NVIDIA Container Toolkit problem
   on the host, not something the `quannet-ssl` image can fix:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi
   ```
2. **Clone the repo:**
   ```bash
   git clone git@github.com:aidartangatov/viscosity-estimator.git
   cd viscosity-estimator/ViscosityEstimator
   ```
   No SSH key registered on this box yet? Clone over HTTPS with a
   fine-grained GitHub personal access token instead:
   `git clone https://<token>@github.com/aidartangatov/viscosity-estimator.git`.
3. **Build the training image** (a few minutes - downloads the ~6GB PyTorch
   CUDA base layer the first time):
   ```bash
   docker build -t quannet-ssl -f docker/ssl/Dockerfile .
   ```
4. **ClearML credentials** - from ClearML → Settings → Workspace → Create
   new credentials, either export them as env vars or mount an existing
   `clearml.conf`:
   ```bash
   export CLEARML_API_HOST=https://api.clear.ml
   export CLEARML_API_ACCESS_KEY=...
   export CLEARML_API_SECRET_KEY=...
   ```
5. **Smoke test first** - a couple of minutes, catches a broken image / CUDA
   mismatch / bad ClearML credentials cheaply, before spending real GPU-hours:
   ```bash
   docker run --gpus all \
     -e CLEARML_API_HOST -e CLEARML_API_ACCESS_KEY -e CLEARML_API_SECRET_KEY \
     quannet-ssl pretrain --method vicreg \
     --clearml-dataset quannet-ssl/igfold_oas_esp_5k \
     --output-dir /tmp/smoke --max-epochs 1 --batch-size 2 \
     --limit-structures 6 --accelerator gpu \
     --clearml-project quannet-ssl --clearml-task-name smoke-test
   ```
6. **Real pretrain run** - see B3 below. Runs for hours; launch it inside
   `tmux`/`screen` (or `docker run -d`) so a dropped SSH session doesn't
   kill it.
7. **Fine-tune / evaluate** - see B4; chain straight off step 6 with
   `--clearml-ssl-checkpoint <pretrain_task_id>`, no manual file copying.
8. **Compare against the baselines** - see B5.
9. **Tear down the instance** - results (metrics, predictions, the trained
   encoder/checkpoint) are already in ClearML as Task artifacts/an
   `OutputModel`, independent of `/app/runs` on the box, so nothing needs to
   be copied off by hand before terminating it.

**VRAM sizing:** the encoder itself is tiny (~130K params). At the default
`--batch-size 8`, VICReg needs roughly 1-2GB of activation memory and MAE
roughly 1.5-2.5GB (its decoder upsamples back to the full 96³ grid, making
it the heavier of the two pretext tasks). 8GB VRAM is a safe minimum, 12-16GB
comfortable if you want to push `--batch-size` higher.

**CUDA/driver mismatch:** `docker/ssl/Dockerfile` is pinned to
`pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`. This works with any host
driver new enough to run CUDA 11.7 (true for essentially all currently-rented
GPUs), but if step 1 or the smoke test throws a CUDA/driver-version error on
unusually new hardware, that base image tag is the first thing to bump.

**Spot/preemptible instances:** `--checkpoint-every-n-epochs` (default 1)
uploads the last Lightning checkpoint to the ClearML task every N epochs
(see B3). If the instance is killed, download `last_checkpoint` from that
task's artifacts on a fresh instance and resume with
`--resume-from-checkpoint`.

## Stage A. Data preprocessing (APBS, CPU)

Turns raw antibody structures (PDB files - either downloaded from SAbDab or
predicted by IgFold, e.g. the OAS-paired archive) into the ESP tensor caches
Stage B reads. Three steps, all CPU-only:

**A1. Build the preprocessing image** (from the `ViscosityEstimator` repo root):

```bash
docker build -t quannet:latest .
```

This bundles APBS 3.4.1 + pdb2pqr; `scripts/build_esp_dataset.py` launches
short-lived containers from it, so it must exist before step A3.

**A2. Get raw PDBs and trim them to a single antibody Fv pair.** Two sources
work out of the box:

```bash
# SAbDab (experimental structures) - downloads + writes manifest.csv
python scripts/download_sabdab.py --out /path/to/raw_pdbs --workers 20

# Extract just the Hchain/Lchain pair per manifest row (drops antigen,
# waters, extra crystal copies - raw SAbDab files are whole crystallographic
# complexes, not a single antibody, and won't fit the ESP grid otherwise)
python scripts/extract_antibody_chains.py \
  --manifest /path/to/raw_pdbs/manifest.csv \
  --source-dir /path/to/raw_pdbs \
  --out /path/to/clean_pdbs \
  --type FV   # keep only Fv-scale entries (FAB/FAB+FC are too large for the 96^3 grid)
```

For a corpus that's already single-antibody-per-file (e.g. IgFold/OAS
predictions, which are Fv-only by construction) skip `extract_antibody_chains.py`
entirely and point step A3 straight at the raw structures directory.

**A3. Build the ESP tensor cache** (one `.npy` per rotation, grouped by structure):

```bash
python scripts/build_esp_dataset.py \
  --structures-dir /path/to/clean_pdbs \
  --artefacts-dir /path/to/esp_cache/artefacts \
  --num-augmentations 5 --processes 8 --batch-size 10
```

Resumable and crash-tolerant by design: a bad structure (malformed PDB,
pdb2pqr failure) is caught per-structure and logged to
`<artefacts-dir's parent>/build_failed.csv` without stopping the rest of the
batch; re-running the same command later skips whatever's already built and
only processes what's missing. `--batch-size` groups structures per Docker
container call to amortize container-startup cost - raise it if you have
many small/fast structures, lower it if a single bad structure should be
isolated faster. `--limit N` caps how many structures to process, for a
quick pilot run before committing to the full corpus.

Do this once per data source (SAbDab, OAS, the labeled `full_dataset`, ...)
into its own `--artefacts-dir`; Stage B accepts multiple cache roots at once.

## Stage B. SSL pretrain + fine-tune (GPU)

### B1. Prerequisites: ESP tensor caches

Both pretext tasks read the on-disk layout produced by `scripts/build_esp_dataset.py`
(Stage A3 above): one subdirectory per antibody structure, containing its
rotation-augmented `*.npy` files. Multiple cache roots can be passed together.

### B2. Build the image

```bash
docker build -t quannet-ssl -f docker/ssl/Dockerfile .
```

Requires a machine with an NVIDIA GPU + the NVIDIA Container Toolkit for
`docker run --gpus all`. There is no CPU-only fallback image for training -
this pipeline is compute-heavy enough that CPU training isn't practical
beyond the tiny debug runs below.

One image, one `ENTRYPOINT` (`scripts/ssl_entrypoint.py`): the first argument
to `docker run` picks the subcommand (`pretrain` or `finetune`), everything
after it is passed straight through to that script's own argparse - no more
`--entrypoint` overrides needed to switch between the two stages.

### B3. Pretrain

The unlabeled corpus already lives in ClearML (project `quannet-ssl`, uploaded
via `scripts/upload_clearml_dataset.py`) - no `-v` mount needed for the data
itself, just for `runs/`:

| Dataset | Structures | Compressed size |
|---|---|---|
| `quannet-ssl/sabdab_esp` | 3574 SAbDab Fv structures | 10.54 GiB |
| `quannet-ssl/igfold_oas_esp_5k` | 2250 IgFold/OAS-predicted Fv structures | 6.54 GiB |
| `quannet-ssl/labeled_esp` | 55/56 labeled antibodies (fine-tune only) | 0.49 GiB |

```bash
docker run --gpus all \
  -v /path/to/runs:/app/runs \
  -e CLEARML_API_HOST=... -e CLEARML_API_ACCESS_KEY=... -e CLEARML_API_SECRET_KEY=... \
  quannet-ssl pretrain \
    --method vicreg \
    --clearml-dataset quannet-ssl/sabdab_esp --clearml-dataset quannet-ssl/igfold_oas_esp_5k \
    --output-dir /app/runs/ssl_vicreg \
    --accelerator gpu --max-epochs 50 --batch-size 32 \
    --clearml-project quannet-ssl --clearml-task-name vicreg-sabdab-oas
```

(`--data-dirs` is optional when `--clearml-dataset` is given - the resolved
local copies of the ClearML datasets are appended to whatever `--data-dirs`
you also pass. Use `--data-dirs` alone, or `--clearml-dataset` alone, or both
together.)

Swap `--method vicreg` for `--method mae` to run the other pretext task
(same data/output-dir conventions, plus `--block-size`/`--mask-ratio` if you
want to override the MAE defaults in `quannet.ssl.mae.MAELitModule`).

Saves `<output-dir>/encoder.pt` - the encoder's `state_dict` only (the
VICReg projector / MAE decoder heads are pretext-task-only and are not
saved, matching how the original papers treat them).

**ClearML integration (all flags optional - omit `--clearml-project` to run
untracked, e.g. for the CPU debug run below):**

- `--clearml-project` / `--clearml-task-name` / `--clearml-tags`: opens a
  ClearML `Task` for this run. Credentials come from the environment
  (`CLEARML_API_HOST`/`CLEARML_API_ACCESS_KEY`/`CLEARML_API_SECRET_KEY`, or a
  mounted `clearml.conf`) - never baked into the image.
- `--clearml-dataset DATASET_PROJECT/DATASET_NAME` (repeatable, or a bare
  dataset ID): pulls a ClearML `Dataset`'s local copy and appends it to
  `--data-dirs`, instead of (or alongside) `-v`-mounted cache directories.
- Live loss curves: every epoch's `train_loss`/`val_loss` (plus VICReg's
  `repr_loss`/`std_loss`/`cov_loss`) are reported straight to the Task's
  SCALARS tab via `quannet.ssl.clearml_utils.ClearMLMetricsLogger` - this is
  independent of whichever logger Lightning happens to pick internally
  (no `tensorboard` package is installed in the image, so relying on
  ClearML's automatic TensorBoard capture would silently show nothing).
  `finetune` reports the same per-fold, tagged `fold_<n>`, in addition to
  the final `spearman`/`mae`/`r2` single-value metrics logged at the end.
- The trained `encoder.pt` is uploaded as a ClearML `OutputModel` named
  `<method>_encoder` when a Task is open, so `finetune`'s
  `--clearml-ssl-checkpoint <task_id>` can pull it directly - no manual file
  copying between pretrain and finetune runs.
- The last Lightning checkpoint is also uploaded as a task artifact every
  `--checkpoint-every-n-epochs` epochs, so a preempted spot/preemptible GPU
  instance doesn't lose progress even if its local disk goes with it
  (`--resume-from-checkpoint` still needs the actual `.ckpt` file - download
  it back from the task's artifacts to resume on a fresh instance).

### Debug run without a GPU

Everything above also runs on CPU with a small `--limit-structures` and
`--batch-size`, e.g.:

```bash
docker run quannet-ssl pretrain --method vicreg \
  --data-dirs /path/to/small_cache --output-dir /tmp/ssl_debug \
  --max-epochs 1 --batch-size 2 --limit-structures 6 --accelerator cpu

# or without Docker, straight through the underlying script:
python -m quannet.ssl.pretrain --method vicreg \
  --data-dirs /path/to/small_cache --output-dir /tmp/ssl_debug \
  --max-epochs 1 --batch-size 2 --limit-structures 6 --accelerator cpu
```

This is how the pipeline was smoke-tested before a GPU server was available -
it validates the full CLI path (dataset loading, model, training loop,
checkpoint saving) in a couple of minutes, not a real pretraining run.

### B4. Fine-tune / evaluate

`scripts/finetune_ssl_resnet.py` runs the same lineage-stratified 5-fold CV
as `scripts/run_cv_resnet.py` (the "without SSL" baseline), initializing the
encoder from a pretraining checkpoint instead of from scratch, and writes
`predictions.csv`/`metrics.json` in the same schema so all three models
(ESM2 baseline, from-scratch ResNet-3D, SSL-pretrained ResNet-3D) are
directly comparable:

```bash
# Full fine-tune (encoder + head both train), labeled ESP cache from ClearML
docker run --gpus all -v /path/to/runs:/app/runs quannet-ssl finetune \
  --clearml-dataset quannet-ssl/labeled_esp \
  --clearml-ssl-checkpoint <pretrain_task_id> \
  --clearml-project quannet-ssl --clearml-task-name finetune-vicreg-sabdab-oas

# Linear probe (encoder frozen, only the regression head trains)
docker run --gpus all -v /path/to/runs:/app/runs quannet-ssl finetune \
  --clearml-dataset quannet-ssl/labeled_esp \
  --clearml-ssl-checkpoint <pretrain_task_id> --freeze_encoder
```

Requires the labeled dataset's own ESP cache (`--cache_root`, default
`runs/train2_fixed/artefacts`, or `--clearml-dataset quannet-ssl/labeled_esp`
- already uploaded, 55/56 labeled antibodies, 3 augmentations each) to
already exist. `--clearml-ssl-checkpoint <task_id>` pulls the encoder
straight from a `pretrain` Task's output model instead of a local
`--ssl_checkpoint` path. On completion it logs `spearman`/`mae`/`r2` as
single-value metrics and uploads `predictions.csv`/`metrics.json` as task
artifacts.

### B5. Comparing against the baselines

| Model | Command | Output |
|---|---|---|
| ESM2 (frozen embeddings + RidgeCV) | `python -m quannet.baselines.esm2.run ...` | `runs/esm2_baseline_lineage/metrics.json` |
| ResNet-3D, no SSL (from scratch) | `python scripts/run_cv_resnet.py` | `runs/resnet3d_cv_lineage/metrics.json` |
| ResNet-3D, SSL-pretrained | `python scripts/finetune_ssl_resnet.py --ssl_checkpoint ...` | `runs/resnet3d_ssl_cv_lineage/metrics.json` |

All three report `spearman`/`mae`/`r2` computed the same way (out-of-fold
predictions over the fixed `datasets/full_dataset/cv_folds.csv` lineage
folds), so the numbers are comparable directly.
