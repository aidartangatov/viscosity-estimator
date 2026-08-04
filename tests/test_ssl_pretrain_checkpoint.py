"""Round-trip test: a checkpoint saved by quannet.ssl.pretrain must load
cleanly into a fresh ResNet3DModule the way scripts/finetune_ssl_resnet.py
does it - this is the seam between the two halves of the SSL pipeline
(pretrain writes the file, fine-tune reads it), so it's worth its own test
independent of either script's internals.
"""
from pathlib import Path

import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quannet.ssl.pretrain import build_module_and_dataset  # noqa: E402
from quannet.models.resnet3d.model import ResNet3DModule  # noqa: E402


def _make_cache(tmp_path, n_structures=4, n_rot=2):
    import numpy as np

    root = tmp_path / 'artefacts'
    for i in range(n_structures):
        d = root / f's{i}'
        d.mkdir(parents=True)
        for r in range(n_rot):
            np.save(d / f's{i}_rot{r}.npy', np.zeros((8, 8, 8), dtype=np.float32))
    return root


def test_pretrained_encoder_loads_into_fresh_module(tmp_path):
    root = _make_cache(tmp_path)
    encoder, module, dataset = build_module_and_dataset('vicreg', [root], grid_dim=8)
    assert len(dataset) == 4

    # Simulate what quannet.ssl.pretrain.main() does after trainer.fit(): save
    # just the encoder's state_dict, not the pretext-task head (projector).
    checkpoint_path = tmp_path / 'encoder.pt'
    torch.save(encoder.state_dict(), checkpoint_path)

    # Simulate what scripts/finetune_ssl_resnet.py does: construct a fresh
    # encoder of the same class and load the checkpoint with strict=True.
    fresh = ResNet3DModule()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    fresh.load_state_dict(state_dict, strict=True)  # must not raise

    # Weights actually transferred, not just shape-compatible zeros.
    for (name, p_pretrained), (_, p_fresh) in zip(encoder.named_parameters(), fresh.named_parameters()):
        assert torch.equal(p_pretrained, p_fresh), f'{name} did not round-trip'
