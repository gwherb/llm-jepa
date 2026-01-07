# GPT2-JEPA DataLoader Documentation

## Overview

The `gpt2_jepa_dataloader.py` module provides PyTorch Dataset and DataLoader implementations for the reversal curse training task with JEPA.

## Data Format

### Training Data (`train.json`)

- **Format**: JSON array containing 390,000 samples
- **Structure**:
  ```json
  [
    {
      "input_text": [[e1_first, e1_last], [relation], [mask]],
      "target_text": [[e2_first, e2_last]],
      "type": "train"
    },
    ...
  ]
  ```

- **Types**:
  - `"train"` (300,000 samples): Both forward and reverse directions exist in training set. JEPA loss is applied.
  - `"atomic"` (90,000 samples): Only one direction in training set. JEPA loss is NOT applied.

### Validation Data (`valid.json`)

- **Format**: Nested JSON with three splits
- **Structure**:
  ```json
  {
    "train": [...],   // Samples with both directions
    "atomic": [...],  // Samples with one direction only
    "test": [...]     // Reverse directions of atomic samples (held out)
  }
  ```

## Usage

### Basic Usage

```python
from models.gpt2_jepa_dataloader import create_dataloaders, get_special_token_ids

# Create dataloaders
train_loader, valid_loader = create_dataloaders(
    train_path='data/inversionidcomb10.50000.30000/train.json',
    valid_path='data/inversionidcomb10.50000.30000/valid.json',
    batch_size=32,
    shuffle_train=True,
    valid_split='train',  # Use 'train', 'atomic', or 'test' split
)

# Get special token IDs
special_tokens = get_special_token_ids('data/inversionidcomb10.50000.30000/vocab.json')
pred_token_id = special_tokens['pred']              # <PRED> token for JEPA
mask_token_id = special_tokens['mask']              # <mask> token
first_relation_id = special_tokens['first_relation']  # <r_0> for relation inversion

# Iterate over batches
for batch in train_loader:
    input_ids = batch['input_ids']  # (batch_size, 6)
    labels = batch['labels']        # (batch_size, 6)
    types = batch['type']           # List of 'train' or 'atomic'
```

### Batch Format

Each batch contains:

- **`input_ids`**: `(batch_size, 6)` tensor
  - Format: `[e1_first, e1_last, relation, mask, e2_first, e2_last]`
  - All 6 tokens are entity/relation/mask tokens

- **`labels`**: `(batch_size, 6)` tensor
  - Format: `[-100, -100, -100, -100, e2_first, e2_last]`
  - Only target entity tokens are supervised (positions 4-5)
  - First 4 positions are masked with -100 (not used in loss)

- **`type`**: List of strings
  - Values: `"train"` or `"atomic"`
  - Used to determine whether to compute JEPA loss

## Integration with Training

### Complete Example

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from models.gpt2_jepa import GPT2WithJEPA, train_jepa
from models.gpt2_jepa_config import JEPAConfig
from models.gpt2_jepa_dataloader import create_dataloaders, get_special_token_ids

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, valid_loader = create_dataloaders(
    train_path='data/inversionidcomb10.50000.30000/train.json',
    valid_path='data/inversionidcomb10.50000.30000/valid.json',
    batch_size=32,
)

special_tokens = get_special_token_ids('data/inversionidcomb10.50000.30000/vocab.json')

# Create model
config = GPT2Config(vocab_size=16015, n_embd=768, n_layer=12, n_head=12)
base_model = GPT2LMHeadModel(config)
jepa_config = JEPAConfig(lambda_jepa=1.0, gamma_ntp=1.0, k_pred_tok=1)
model = GPT2WithJEPA(base_model=base_model, jepa_config=jepa_config)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

training_stats = train_jepa(
    model=model,
    train_dataloader=train_loader,
    optimizer=optimizer,
    num_epochs=3,
    device=device,
    pred_token_id=special_tokens['pred'],
    first_relation_token_id=special_tokens['first_relation'],
    eval_dataloader=valid_loader,
    save_path='checkpoints/',
)
```

## Dataset Class

### `ReversalCurseDataset`

Custom PyTorch Dataset for loading reversal curse data.

**Constructor**:
```python
dataset = ReversalCurseDataset(
    data_path='train.json',
    data_key=None,  # None for train.json, 'train'/'atomic'/'test' for valid.json
)
```

**Methods**:
- `__len__()`: Returns number of samples
- `__getitem__(idx)`: Returns dict with `input_ids`, `labels`, and `type`

## Utility Functions

### `create_dataloaders()`

Creates train and validation DataLoaders.

**Parameters**:
- `train_path` (str): Path to training JSON file
- `valid_path` (str): Path to validation JSON file
- `batch_size` (int): Batch size (default: 32)
- `num_workers` (int): Number of workers for DataLoader (default: 0)
- `shuffle_train` (bool): Whether to shuffle training data (default: True)
- `valid_split` (str): Validation split to use - 'train', 'atomic', or 'test' (default: 'train')

**Returns**: `(train_dataloader, valid_dataloader)`

### `get_special_token_ids()`

Extracts special token IDs from vocabulary.

**Parameters**:
- `vocab_path` (str): Path to vocab.json

**Returns**: Dict with keys `'pred'`, `'mask'`, `'first_relation'`

## Notes

1. **Sequence Length**: All sequences are exactly 6 tokens long
2. **Label Masking**: Only target entity tokens (last 2 positions) are supervised
3. **Type Field**: The `type` field is crucial for selective JEPA loss computation
4. **Reverse Generation**: Reverse sequences are generated automatically during training using `GPT2WithJEPA.create_reverse_sequence()`

## Testing

Run the module directly to test functionality:

```bash
python models/gpt2_jepa_dataloader.py
```

This will:
1. Load and validate training data format
2. Load and validate validation data splits
3. Test batch creation and collation
4. Display special token IDs
