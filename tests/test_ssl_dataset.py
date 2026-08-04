from pathlib import Path

import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quannet.ssl.dataset import MAEPretrainDataset, VICRegPretrainDataset, list_structure_npy_groups  # noqa: E402


def _make_cache(tmp_path, structures):
    """structures: dict of {structure_name: n_rotations}"""
    root = tmp_path / 'artefacts'
    for name, n_rot in structures.items():
        d = root / name
        d.mkdir(parents=True)
        for i in range(n_rot):
            np.save(d / f'{name}_rot{i}.npy', np.zeros((4, 4, 4), dtype=np.float32))
    return root


def test_list_structure_npy_groups_across_multiple_roots(tmp_path):
    root1 = _make_cache(tmp_path / 'a', {'s1': 5, 's2': 5})
    root2 = _make_cache(tmp_path / 'b', {'s3': 3})
    groups = list_structure_npy_groups([root1, root2])
    assert len(groups) == 3
    assert sorted(len(g) for g in groups) == [3, 5, 5]


def test_list_structure_npy_groups_skips_missing_root(tmp_path):
    root1 = _make_cache(tmp_path / 'a', {'s1': 5})
    groups = list_structure_npy_groups([root1, tmp_path / 'does_not_exist'])
    assert len(groups) == 1


def test_vicreg_dataset_yields_two_distinct_views(tmp_path):
    root = _make_cache(tmp_path, {'s1': 5, 's2': 5})
    ds = VICRegPretrainDataset([root])
    assert len(ds) == 2
    x1, x2 = ds[0]
    assert x1.shape == (1, 4, 4, 4)
    assert x2.shape == (1, 4, 4, 4)


def test_vicreg_dataset_skips_structures_with_too_few_views(tmp_path):
    root = _make_cache(tmp_path, {'s1': 1, 's2': 5})
    ds = VICRegPretrainDataset([root], min_views=2)
    assert len(ds) == 1


def test_mae_dataset_flattens_all_rotations(tmp_path):
    root = _make_cache(tmp_path, {'s1': 5, 's2': 3})
    ds = MAEPretrainDataset([root])
    assert len(ds) == 8
    x = ds[0]
    assert x.shape == (1, 4, 4, 4)
