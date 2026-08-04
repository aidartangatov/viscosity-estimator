"""
Host-side supervisor around `run_make_inputs` that processes structures in
small batches so a single bad PDB (pdb2pqr crash, APBS mesh error, etc.)
can't take down an entire multi-hour run, while still amortizing the
per-container startup cost across several structures.

`quannet.make_inputs.make_inputs` (container-side) now catches per-structure
exceptions internally and moves on, so one bad structure inside a batch
leaves its batch-mates unaffected. This script still verifies each
structure's output directory on disk after every batch (rather than trusting
the container's exit status), and skips structures that are already fully
built, so an interrupted run can simply be re-invoked. Observed failure rate
on a real SAbDab pilot batch was ~3.4% (pdb2pqr chokes on certain
backbone-gap patterns with "Unknown format code 'i' for object of type
'int'").

Usage:
    python scripts/build_esp_dataset.py \\
        --structures-dir "C:\\Users\\admin\\PycharmProjects\\data\\antibody_sabdab_fv_pilot" \\
        --artefacts-dir  "C:\\Users\\admin\\PycharmProjects\\data\\sabdab_esp_pilot\\artefacts" \\
        --num-augmentations 5 --processes 8 --batch-size 10
"""

from pathlib import Path

import csv
import sys
import time
import shutil
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quannet.utils import InsufficientDataError  # noqa: E402
from quannet.docker.make_inputs import run_make_inputs  # noqa: E402


def is_done(output_dir: Path, num_augmentations: int) -> bool:
    return output_dir.is_dir() and len(list(output_dir.glob("*.npy"))) >= num_augmentations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures-dir", required=True)
    ap.add_argument("--artefacts-dir", required=True)
    ap.add_argument("--num-augmentations", type=int, default=5)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--image", default="quannet:latest")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N structures")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="structures per docker container call. Bigger batches amortize container "
        "startup cost; a bad structure inside a batch is caught and skipped in-container, "
        "it does not affect its batch-mates.",
    )
    args = ap.parse_args()

    structures_dir = Path(args.structures_dir)
    artefacts_dir = Path(args.artefacts_dir)
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    structure_paths = sorted(structures_dir.glob("*.pdb"))
    if args.limit:
        structure_paths = structure_paths[: args.limit]
    print(f"{len(structure_paths)} structures found in {structures_dir}", flush=True)

    fail_log_path = artefacts_dir.parent / "build_failed.csv"
    fail_rows = []
    n_ok = n_skip = n_fail = 0
    n_done = 0
    t0 = time.perf_counter()

    pending = []
    for structure_path in structure_paths:
        stem = structure_path.with_suffix("").name
        output_dir = artefacts_dir / stem
        if is_done(output_dir, args.num_augmentations):
            n_skip += 1
            n_done += 1
        else:
            if output_dir.exists():
                shutil.rmtree(output_dir)  # partial/failed leftovers from a previous crash
            pending.append(structure_path)

    batches = [pending[i : i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    print(
        f"{n_skip} already built, {len(pending)} pending in {len(batches)} batches of <= {args.batch_size}", flush=True
    )

    for batch in batches:
        try:
            run_make_inputs(
                structure_paths=batch,
                artefacts_dir=artefacts_dir,
                remove_artefacts=True,
                train_mode=True,
                num_augmentations=args.num_augmentations,
                processes=args.processes,
                image=args.image,
                collect_arrays=False,
            )
        except (InsufficientDataError, ValueError) as e:
            # The whole container call failed (e.g. docker daemon hiccup) - every
            # structure in this batch gets checked individually below regardless.
            print(f"[batch] container call raised {type(e).__name__}: {e}", flush=True)

        for structure_path in batch:
            stem = structure_path.with_suffix("").name
            output_dir = artefacts_dir / stem
            if is_done(output_dir, args.num_augmentations):
                n_ok += 1
            else:
                n_fail += 1
                fail_rows.append((stem, "IncompleteOutput", f"expected {args.num_augmentations} .npy files"))
                if output_dir.exists():
                    shutil.rmtree(output_dir)

        n_done += len(batch)
        elapsed = time.perf_counter() - t0
        rate = n_done / elapsed
        eta_min = (len(structure_paths) - n_done) / rate / 60 if rate > 0 else 0
        print(
            f"[{n_done}/{len(structure_paths)}] ok={n_ok} skip={n_skip} fail={n_fail} "
            f"| {rate:.2f} struct/s | ETA {eta_min:.1f} min",
            flush=True,
        )

    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min. ok={n_ok} skip={n_skip} fail={n_fail}")
    if fail_rows:
        with open(fail_log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["structure", "error_type", "error_message"])
            w.writerows(fail_rows)
        print(f"Failure log: {fail_log_path}")


if __name__ == "__main__":
    main()
