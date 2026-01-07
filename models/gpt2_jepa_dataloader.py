"""
DataLoader for GPT2-JEPA Reversal Curse Training

This module provides dataset and dataloader utilities for the reversal curse task.
It handles the nested JSON format and creates proper input/label tensors for training.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional


class ReversalCurseDataset(Dataset):
    """
    Dataset for reversal curse training with JEPA.

    Input format (train.json is a JSON array):
        [{"input_text": [[e1_f, e1_l], [r], [mask]], "target_text": [[e2_f, e2_l]], "type": "train"}, ...]

    Output format:
        - input_ids: [e1_f, e1_l, r, mask, e2_f, e2_l] (6 tokens)
        - labels: [-100, -100, -100, -100, e2_f, e2_l] (only target tokens supervised)
        - type: "train" or "atomic"
    """

    def __init__(
        self,
        data_path: str,
        data_key: Optional[str] = None,
    ):
        """
        Args:
            data_path: Path to JSON file (train.json or valid.json)
            data_key: Optional key to access nested data (e.g., "train", "atomic", "test")
                     If None, assumes JSON array format (training data)
        """
        self.data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

            if data_key is not None:
                # Nested format (validation data)
                self.data = loaded_data[data_key]
            else:
                # Direct array format (training data)
                if isinstance(loaded_data, list):
                    self.data = loaded_data
                else:
                    raise ValueError(f"Expected list in {data_path}, got {type(loaded_data)}")

        print(f"Loaded {len(self.data)} samples from {data_path}" +
              (f" (key: {data_key})" if data_key else ""))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a single training example.

        Returns:
            Dictionary with:
                - input_ids: List of 6 token IDs
                - labels: List of 6 token IDs (first 4 are -100)
                - type: "train" or "atomic"
        """
        sample = self.data[idx]

        # Parse nested structure
        # input_text: [[e1_f, e1_l], [r], [mask]]
        # target_text: [[e2_f, e2_l]]
        input_text = sample['input_text']
        target_text = sample['target_text']
        sample_type = sample['type']

        # Flatten input: [e1_f, e1_l] + [r] + [mask] = [e1_f, e1_l, r, mask]
        e1_tokens = input_text[0]  # [e1_f, e1_l]
        r_token = input_text[1]    # [r]
        mask_token = input_text[2] # [mask]

        # Flatten target: [e2_f, e2_l]
        e2_tokens = target_text[0]  # [e2_f, e2_l]

        # Construct full sequence: [e1_f, e1_l, r, mask, e2_f, e2_l]
        input_ids = e1_tokens + r_token + mask_token + e2_tokens

        # Create labels: only supervise target tokens (last 2 positions)
        # [-100, -100, -100, -100, e2_f, e2_l]
        labels = [-100, -100, -100, -100] + e2_tokens

        return {
            'input_ids': input_ids,
            'labels': labels,
            'type': sample_type,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to batch samples together.

    Args:
        batch: List of samples from ReversalCurseDataset

    Returns:
        Dictionary with:
            - input_ids: (batch_size, 6) tensor
            - labels: (batch_size, 6) tensor
            - type: List of strings
    """
    input_ids = torch.tensor([sample['input_ids'] for sample in batch], dtype=torch.long)
    labels = torch.tensor([sample['labels'] for sample in batch], dtype=torch.long)
    types = [sample['type'] for sample in batch]

    return {
        'input_ids': input_ids,
        'labels': labels,
        'type': types,
    }


def create_dataloaders(
    train_path: str,
    valid_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    valid_split: str = 'train',
) -> tuple:
    """
    Create training and validation dataloaders.

    Args:
        train_path: Path to training data (JSON array format)
        valid_path: Path to validation data (nested JSON format)
        batch_size: Batch size for both train and valid
        num_workers: Number of workers for DataLoader
        shuffle_train: Whether to shuffle training data
        valid_split: Which split to use from validation data ('train', 'atomic', or 'test')

    Returns:
        (train_dataloader, valid_dataloader) tuple
    """
    # Create datasets
    train_dataset = ReversalCurseDataset(train_path, data_key=None)
    valid_dataset = ReversalCurseDataset(valid_path, data_key=valid_split)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader


def get_special_token_ids(vocab_path: str) -> Dict[str, int]:
    """
    Extract special token IDs from vocabulary file.

    Args:
        vocab_path: Path to vocab.json

    Returns:
        Dictionary with special token IDs:
            - 'pred': <PRED> token ID
            - 'mask': <mask> token ID
            - 'first_relation': <r_0> token ID
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    return {
        'pred': vocab['<PRED>'],
        'mask': vocab['<mask>'],
        'first_relation': vocab['<r_0>'],
    }


if __name__ == '__main__':
    """Example usage and testing"""
    import os

    # Paths
    base_path = r'c:\Users\gwher\OneDrive\Desktop\llm-jepa\data\inversionidcomb10.50000.30000'
    train_path = os.path.join(base_path, 'train.json')
    valid_path = os.path.join(base_path, 'valid.json')
    vocab_path = os.path.join(base_path, 'vocab.json')

    print("Testing ReversalCurseDataset...")
    print("="*70)

    # Test training dataset
    print("\n1. Training Dataset (JSONL format):")
    train_dataset = ReversalCurseDataset(train_path, data_key=None)
    print(f"   Total samples: {len(train_dataset)}")
    print(f"   Sample 0: {train_dataset[0]}")
    print(f"   Input IDs shape: {len(train_dataset[0]['input_ids'])}")
    print(f"   Labels shape: {len(train_dataset[0]['labels'])}")

    # Test validation dataset
    print("\n2. Validation Dataset (nested JSON format):")
    valid_dataset_train = ReversalCurseDataset(valid_path, data_key='train')
    valid_dataset_atomic = ReversalCurseDataset(valid_path, data_key='atomic')
    valid_dataset_test = ReversalCurseDataset(valid_path, data_key='test')
    print(f"   'train' split: {len(valid_dataset_train)} samples")
    print(f"   'atomic' split: {len(valid_dataset_atomic)} samples")
    print(f"   'test' split: {len(valid_dataset_test)} samples")
    print(f"   Sample from 'train': {valid_dataset_train[0]}")

    # Test dataloader
    print("\n3. DataLoader with batching:")
    train_loader, valid_loader = create_dataloaders(
        train_path, valid_path, batch_size=4, valid_split='train'
    )

    batch = next(iter(train_loader))
    print(f"   Batch keys: {batch.keys()}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   types: {batch['type']}")
    print(f"   Sample input_ids: {batch['input_ids'][0]}")
    print(f"   Sample labels: {batch['labels'][0]}")

    # Test special tokens
    print("\n4. Special Token IDs:")
    special_tokens = get_special_token_ids(vocab_path)
    print(f"   <PRED>: {special_tokens['pred']}")
    print(f"   <mask>: {special_tokens['mask']}")
    print(f"   <r_0>: {special_tokens['first_relation']}")

    print("\n" + "="*70)
    print("All tests passed!")
