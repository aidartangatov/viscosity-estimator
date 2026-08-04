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

```bash
docker run --gpus all \
  -v /path/to/esp_caches:/data:ro \
  -v /path/to/runs:/app/runs \
  -e CLEARML_API_HOST=... -e CLEARML_API_ACCESS_KEY=... -e CLEARML_API_SECRET_KEY=... \
  quannet-ssl pretrain \
    --method vicreg \
    --data-dirs /data/sabdab_esp/artefacts /data/oas_esp/artefacts \
    --output-dir /app/runs/ssl_vicreg \
    --accelerator gpu --max-epochs 50 --batch-size 32 \
    --clearml-project quannet-ssl --clearml-task-name vicreg-sabdab-oas
```

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
# Full fine-tune (encoder + head both train)
docker run --gpus all -v /path/to/runs:/app/runs quannet-ssl finetune \
  --ssl_checkpoint /app/runs/ssl_vicreg/encoder.pt
# ...or pull the encoder straight from the pretrain Task instead of a local path:
docker run --gpus all -v /path/to/runs:/app/runs quannet-ssl finetune \
  --clearml-ssl-checkpoint <pretrain_task_id>

# Linear probe (encoder frozen, only the regression head trains)
docker run --gpus all -v /path/to/runs:/app/runs quannet-ssl finetune \
  --ssl_checkpoint /app/runs/ssl_vicreg/encoder.pt --freeze_encoder
```

Requires the labeled dataset's own ESP cache (`--cache_root`, default
`runs/train2_fixed/artefacts`) to already exist - build it the same way as
the pretraining caches, from `datasets/full_dataset/full_dataset/*.pdb`.
Accepts the same `--clearml-project`/`--clearml-dataset` flags as `pretrain`
(the latter overrides `--cache_root` when given); on completion it logs
`spearman`/`mae`/`r2` as single-value metrics and uploads `predictions.csv`/
`metrics.json` as task artifacts.

### B5. Comparing against the baselines

| Model | Command | Output |
|---|---|---|
| ESM2 (frozen embeddings + RidgeCV) | `python -m quannet.baselines.esm2.run ...` | `runs/esm2_baseline_lineage/metrics.json` |
| ResNet-3D, no SSL (from scratch) | `python scripts/run_cv_resnet.py` | `runs/resnet3d_cv_lineage/metrics.json` |
| ResNet-3D, SSL-pretrained | `python scripts/finetune_ssl_resnet.py --ssl_checkpoint ...` | `runs/resnet3d_ssl_cv_lineage/metrics.json` |

All three report `spearman`/`mae`/`r2` computed the same way (out-of-fold
predictions over the fixed `datasets/full_dataset/cv_folds.csv` lineage
folds), so the numbers are comparable directly.
