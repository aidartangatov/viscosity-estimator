from typing import Dict, Tuple
from Bio.PDB import PDBParser
from pathlib import Path
from Bio.PDB.Polypeptide import is_aa, three_to_one

HEAVY_LIGHT = ('H', 'L')


def extract_chain_sequences(pdb_path: Path, chain_ids: Tuple[str, ...] = HEAVY_LIGHT) -> Dict[str, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    sequences: Dict[str, str] = {}
    for model in structure:
        for chain in model:
            if chain.id not in chain_ids:
                continue
            seq_chars = []
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                try:
                    seq_chars.append(three_to_one(residue.get_resname()))
                except KeyError:
                    continue
            sequences[chain.id] = ''.join(seq_chars)
        break
    return sequences
