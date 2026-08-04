"""
Extract single antibody (Hchain + Lchain) copies from raw SAbDab crystal
structures, so they can be fed into the APBS/ESP pipeline the same way the
IgFold-predicted `full_dataset` structures are.

Raw SAbDab downloads are whole crystallographic asymmetric units: they may
carry an antigen, waters, sugars, and/or multiple independent copies of the
antibody. Feeding those directly to APBS overflows the 96^3 @ 0.75A grid
(centered on the whole complex) and silently drops atoms outside the box.
This script trims each manifest row down to just its Hchain/Lchain pair.

Usage:
    python scripts/extract_antibody_chains.py
    python scripts/extract_antibody_chains.py --limit 50 --out datasets/Sabdab/structures
"""

from tqdm import tqdm
from Bio.PDB import PDBIO, Select, PDBParser
from pathlib import Path

import csv
import pandas as pd
import argparse

DEFAULT_MANIFEST = r"C:\Users\admin\PycharmProjects\data\antibody_sabdab\manifest.csv"
DEFAULT_SOURCE_DIR = r"C:\Users\admin\PycharmProjects\data\antibody_sabdab"
DEFAULT_OUT_DIR = "datasets/Sabdab/structures"


class TwoChainSelect(Select):
    def __init__(self, keep):
        self.keep = set(keep)

    def accept_chain(self, chain):
        return chain.id in self.keep


def is_ca_only(pdb_path: Path) -> bool:
    with pdb_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("MDLTYP"):
                return "CA ATOMS ONLY" in line
            if line.startswith(("ATOM", "HETATM")):
                break
    return False


def extract_one(parser, io, row, source_dir: Path, out_dir: Path):
    pdb_path = source_dir / f"{row.pdb_code}.pdb"
    if not pdb_path.exists():
        return "missing_source"
    if len(row.Hchain) != 1 or len(row.Lchain) != 1:
        return "multi_char_chain_id"
    if is_ca_only(pdb_path):
        return "ca_only"

    structure = parser.get_structure(row.pdb_code, pdb_path)
    try:
        model = next(iter(structure))
    except StopIteration:
        return "no_models"

    chain_ids = {c.id for c in model}
    if row.Hchain not in chain_ids or row.Lchain not in chain_ids:
        return "chain_not_found"

    dest = out_dir / f"{row.INSTANCE}.pdb"
    io.set_structure(model)
    io.save(str(dest), TwoChainSelect([row.Hchain, row.Lchain]))

    n_atoms = sum(1 for c in model for _ in c.get_atoms() if c.id in {row.Hchain, row.Lchain})
    if n_atoms == 0:
        dest.unlink(missing_ok=True)
        return "empty_selection"
    return "ok"


def main():
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser_arg.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser_arg.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser_arg.add_argument("--limit", type=int, default=None, help="only process the first N rows (smoke test)")
    parser_arg.add_argument(
        "--type",
        default=None,
        help="only process manifest rows with this 'type' value (e.g. FV). Default: no filter.",
    )
    args = parser_arg.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, dtype={"Hchain": str, "Lchain": str})
    if args.type:
        df = df[df["type"] == args.type]
    if args.limit:
        df = df.head(args.limit)
    print(f"Processing {len(df)} manifest rows -> {out_dir}")

    pdb_parser = PDBParser(QUIET=True)
    pdb_io = PDBIO()

    counts = {}
    log_rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        status = extract_one(pdb_parser, pdb_io, row, source_dir, out_dir)
        counts[status] = counts.get(status, 0) + 1
        if status != "ok":
            log_rows.append((row.INSTANCE, row.pdb_code, row.Hchain, row.Lchain, status))

    print("\nDone.")
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status}: {n}")

    if log_rows:
        log_path = out_dir.parent / "extract_skipped.csv"
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance", "pdb_code", "hchain", "lchain", "status"])
            w.writerows(log_rows)
        print(f"  Skipped-row log: {log_path}")


if __name__ == "__main__":
    main()
