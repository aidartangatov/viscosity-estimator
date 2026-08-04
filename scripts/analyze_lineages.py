"""Exploratory: measure pairwise sequence identity across the 56 labeled antibodies
to decide what a 'lineage' (CV group) should be. Not part of the training pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd

HEAVY_LIGHT = ('H', 'L')
THREE_TO_ONE = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
}


def extract_chain_sequences(pdb_path):
    """Minimal dependency-free PDB → {chain_id: seq} using one CA per residue."""
    seqs = {c: [] for c in HEAVY_LIGHT}
    seen = {c: set() for c in HEAVY_LIGHT}
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith('ATOM'):
            continue
        if line[12:16].strip() != 'CA':
            continue
        chain = line[21]
        if chain not in HEAVY_LIGHT:
            continue
        resseq = line[22:27]  # resseq + insertion code
        if resseq in seen[chain]:
            continue
        seen[chain].add(resseq)
        aa = THREE_TO_ONE.get(line[17:20].strip())
        if aa:
            seqs[chain].append(aa)
    return {c: ''.join(v) for c, v in seqs.items()}


DATA = Path(__file__).resolve().parents[1] / 'datasets' / 'full_dataset'
df = pd.read_csv(DATA / 'dataset.csv').drop_duplicates('Entity')

seqs = {}
for ent in df['Entity']:
    s = extract_chain_sequences(DATA / 'full_dataset' / f'{ent}.pdb')
    if all(s.get(c) for c in HEAVY_LIGHT):
        seqs[ent] = s['H'] + '/' + s['L']
entities = list(seqs)
print(f'entities with H+L: {len(entities)}')


def identity(a: str, b: str) -> float:
    # chains separated by '/'; compare per-chain by position (same length variants),
    # fall back to length-normalized Hamming on the shorter length.
    ha, la = a.split('/')
    hb, lb = b.split('/')
    tot = match = 0
    for x, y in ((ha, hb), (la, lb)):
        n = min(len(x), len(y))
        m = max(len(x), len(y))
        match += sum(1 for i in range(n) if x[i] == y[i])
        tot += m
    return match / tot if tot else 0.0


n = len(entities)
M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        M[i, j] = identity(seqs[entities[i]], seqs[entities[j]])


def series(e):
    if e.startswith('R1'):
        return 'R1'
    if e.startswith('R2'):
        return 'R2'
    if e in ('AB-001', 'TGN1412'):
        return e
    return 'therapeutic'


ser = [series(e) for e in entities]
groups = sorted(set(ser))
print('\n=== mean within / cross-series identity ===')
for g in groups:
    idx = [i for i in range(n) if ser[i] == g]
    if len(idx) > 1:
        vals = [M[i, j] for a, i in enumerate(idx) for j in idx[a + 1 :]]
        print(
            f'{g:12s} n={len(idx):2d}  within-identity '
            f'mean={np.mean(vals):.3f} min={np.min(vals):.3f} max={np.max(vals):.3f}'
        )
    else:
        print(f'{g:12s} n={len(idx):2d}  (singleton)')

# cross-series R1 vs R2 vs therapeutic
for a in range(len(groups)):
    for b in range(a + 1, len(groups)):
        ia = [i for i in range(n) if ser[i] == groups[a]]
        ib = [i for i in range(n) if ser[i] == groups[b]]
        vals = [M[i, j] for i in ia for j in ib]
        print(f'cross {groups[a]:12s} vs {groups[b]:12s} ' f'mean={np.mean(vals):.3f} max={np.max(vals):.3f}')


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


# greedy clustering at several thresholds
print('\n=== greedy single-linkage clusters by identity threshold ===')
for thr in (0.999, 0.99, 0.97, 0.95, 0.90, 0.80):
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] >= thr:
                parent[find(parent, i)] = find(parent, j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(parent, i), []).append(entities[i])
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print(f'thr={thr:.3f}: {len(clusters)} clusters, sizes={sizes}')
