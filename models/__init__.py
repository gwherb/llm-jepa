"""
GPT2-JEPA Models Package

This package contains the JEPA implementation for GPT2.
"""

from .gpt2_jepa import GPT2WithJEPA, train_jepa, evaluate_jepa
from .gpt2_jepa_config import JEPAConfig, GPT2JEPAConfig
from .gpt2_jepa_dataloader import (
    ReversalCurseDataset,
    create_dataloaders,
    get_special_token_ids,
    collate_fn,
)

__all__ = [
    'GPT2WithJEPA',
    'train_jepa',
    'evaluate_jepa',
    'JEPAConfig',
    'GPT2JEPAConfig',
    'ReversalCurseDataset',
    'create_dataloaders',
    'get_special_token_ids',
    'collate_fn',
]
