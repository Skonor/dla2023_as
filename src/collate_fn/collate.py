import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    
    audio_length = []

    for ds in dataset_items:
        audio_length.append(ds['audio'].shape[-1])

    batch_audio = torch.zeros(len(audio_length), max(audio_length))
    audio_path = []
    is_spoofed = []

    for i, ds in enumerate(dataset_items):
        batch_audio[i, :audio_length[i]] = ds['audio']
        audio_path.append(ds['audio_path'])
        is_spoofed.append(ds['is_spoofed'])


    audio_length = torch.tensor(audio_length).long()
    is_spoofed = torch.tensor(is_spoofed).long()

    return {
        'audio': batch_audio,
        'audio_length': audio_length,
        'audio_path': audio_path,
        'is_spoofed': is_spoofed
    }
