from typing import Dict, List, Optional
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from quannet.utils import LOGGER

import torch

DEFAULT_MODEL = 'facebook/esm2_t33_650M_UR50D'


@torch.no_grad()
def embed_sequences(
    sequences: Dict[str, str],
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    batch_size: int = 1,
) -> Dict[str, torch.Tensor]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOGGER.info(f'Loading {model_name} on {device} ...')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    keys: List[str] = list(sequences.keys())
    embeddings: Dict[str, torch.Tensor] = {}

    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i : i + batch_size]
        batch_seqs = [sequences[k] for k in batch_keys]
        inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True, add_special_tokens=True).to(device)
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, D)
        # Mean-pool over real tokens, excluding the leading CLS and trailing EOS
        # by zeroing out positions 0 and the last attended position per sample.
        mask = inputs['attention_mask'].clone()
        mask[:, 0] = 0
        eos_idx = inputs['attention_mask'].sum(dim=1) - 1
        mask[torch.arange(mask.size(0)), eos_idx] = 0
        mask = mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        for k, vec in zip(batch_keys, pooled.cpu()):
            embeddings[k] = vec
        LOGGER.info(f'Embedded {i + len(batch_keys)}/{len(keys)}')

    return embeddings


def save_embeddings(embeddings: Dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)


def load_embeddings(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location='cpu')
