"""Ad-hoc timing harness for the PDB -> PQR -> APBS -> ESP pipeline.

Mirrors `quannet.make_inputs.get_esp_array` step by step but wraps each phase
in a perf_counter so we can see where the wall-clock goes. Meant to be run
INSIDE the `quannet:latest` container (APBS/pdb2pqr + env vars live there).

Usage (inside container):
    python3 /work/time_pipeline.py /data/Adalimumab.pdb /tmp/out
"""
from pathlib import Path
from quannet.make_inputs import APBSWrapper, euler_rotate, load_molecule, save_molecule

import sys
import time
import numpy as np
import subprocess

GRID_DIM = 96
GRID_SPACING = 0.75
SHELL_WIDTH = 2.0
ROTATIONS = (37.0, 88.0, 142.0)  # fixed angles so the run is reproducible


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label:<28} {dt:8.3f} s", flush=True)
    return out, dt


def main(structure_path, output_dir):
    structure_path = Path(structure_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    structure, timings["load_molecule"] = timed("load_molecule (parse PDB)", lambda: load_molecule(structure_path))
    n_atoms = sum(1 for _ in structure.get_atoms())
    print(f"  atoms parsed: {n_atoms}", flush=True)

    structure, timings["euler_rotate"] = timed("euler_rotate (rotate atoms)", lambda: euler_rotate(structure, ROTATIONS))

    rotated_path = output_dir / (structure_path.stem + "_rot.pdb")
    _, timings["save_molecule"] = timed("save_molecule (write PDB)", lambda: save_molecule(structure, rotated_path))

    zap = APBSWrapper(
        input_path=rotated_path,
        output_dir=output_dir,
        config_file_name=rotated_path.with_suffix(".in").name,
        grid_dim=GRID_DIM,
        grid_spacing=GRID_SPACING,
        shell_width=SHELL_WIDTH,
        inner_dielectric=2.0,
        outer_dielectric=80,
        logger=None,
    )

    # --- replicate APBSWrapper.run() body, but timed in pieces ---
    _, timings["pdb2pqr"] = timed("pdb2pqr (charges+radii)", zap._run_pdb2pqr)
    _, timings["prepare_config"] = timed("prepare APBS config", zap._prepare_config_file)

    def _run_apbs():
        cmd = [zap.apbs, str(zap.artefacts_paths["config"])]
        with zap.artefacts_paths["log"].open("w") as logf:
            subprocess.run(cmd, stdout=logf, check=True)

    _, timings["apbs"] = timed("APBS (Poisson-Boltzmann)", _run_apbs)
    zap.has_run = True

    _, timings["load_potential"] = timed("load potential .dx", lambda: zap.load_network("potential"))
    _, timings["load_accessibility"] = timed("load accessibility .dx", lambda: zap.load_network("accessibility"))
    # process_array triggers numba JIT compile on first call in this process.
    esp, timings["process_array"] = timed("process_array (shell mask)*", zap.process_array)

    print("-" * 48, flush=True)
    print(f"ESP tensor shape: {esp.shape}, nonzero: {np.count_nonzero(esp)}", flush=True)
    total = sum(timings.values())
    print(f"{'TOTAL (one rotation)':<28} {total:8.3f} s", flush=True)
    print("* process_array includes one-time numba JIT compilation.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
