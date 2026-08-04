"""
Download PDB structure files from RCSB for SAbDab entries filtered by organism.

Usage:
    python scripts/download_sabdab.py
    python scripts/download_sabdab.py --workers 20 --out datasets/Sabdab/data

Skips already-downloaded files. Writes a manifest CSV on completion.
"""

from pathlib import Path
from concurrent.futures import as_completed, ThreadPoolExecutor

import csv
import sys
import time
import pandas as pd
import argparse
import requests

RCSB_URL = "https://files.rcsb.org/download/{code}.pdb"
TARGET_ORGANISMS = {"homo sapiens", "mus musculus"}
DATASETS = [
    "datasets/Sabdab/FAV_dataset.csv",
    "datasets/Sabdab/FV_dataset.csv",
    "datasets/Sabdab/FAB_FC_dataset.csv",
]


def extract_pdb_code(pdb_id: str) -> str:
    """'pdb_000010gh' -> '10gh'"""
    return pdb_id.replace("pdb_", "")[-4:]


def load_filtered_df(dataset_paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in dataset_paths:
        if not Path(p).exists():
            print(f"[WARN] not found: {p}", file=sys.stderr)
            continue
        df = pd.read_csv(p)
        df["source"] = Path(p).stem
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    mask = combined["organism"].str.lower().isin(TARGET_ORGANISMS)
    filtered = combined[mask].copy()
    filtered["pdb_code"] = filtered["PDB"].map(extract_pdb_code)
    return filtered


def download_one(code: str, out_dir: Path, session: requests.Session, timeout: int = 30) -> tuple[str, str]:
    """Returns (code, status) where status is 'ok', 'skip', or error message."""
    dest = out_dir / f"{code}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return code, "skip"
    url = RCSB_URL.format(code=code.upper())
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return code, "ok"
        else:
            return code, f"http_{r.status_code}"
    except Exception as e:
        return code, f"error_{type(e).__name__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16, help="parallel download threads")
    parser.add_argument("--out", default="datasets/Sabdab/data", help="output directory for .pdb files")
    parser.add_argument("--timeout", type=int, default=30, help="per-request timeout in seconds")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and filtering datasets...")
    df = load_filtered_df(DATASETS)
    unique_codes = sorted(df["pdb_code"].unique())
    print(f"  {len(df)} rows across sources -> {len(unique_codes)} unique PDB codes")
    print(f"  Organisms: {sorted(df['organism'].unique())[:5]} ...")

    # Save filtered manifest
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"  Manifest saved: {manifest_path}")

    # Download
    print(f"\nDownloading to {out_dir}/ with {args.workers} workers...")
    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed = []

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("https://", adapter)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, code, out_dir, session, args.timeout): code for code in unique_codes}
        for i, fut in enumerate(as_completed(futures), 1):
            code, status = fut.result()
            if status == "ok":
                counts["ok"] += 1
            elif status == "skip":
                counts["skip"] += 1
            else:
                counts["fail"] += 1
                failed.append((code, status))

            if i % 100 == 0 or i == len(unique_codes):
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (len(unique_codes) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{len(unique_codes)}] "
                    f"ok={counts['ok']} skip={counts['skip']} fail={counts['fail']} "
                    f"| {rate:.1f} req/s | ETA {remaining/60:.1f} min"
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"  Downloaded: {counts['ok']}")
    print(f"  Skipped (already exist): {counts['skip']}")
    print(f"  Failed: {counts['fail']}")

    if failed:
        fail_path = out_dir / "failed.csv"
        with open(fail_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pdb_code", "status"])
            w.writerows(failed)
        print(f"  Failed list: {fail_path}")


if __name__ == "__main__":
    main()
