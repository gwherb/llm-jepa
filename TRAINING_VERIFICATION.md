# LLM-JEPA Training Process Verification

This document verifies that all training requirements are correctly implemented.

## ✅ Requirement 1: First Relation Token for Accurate Reversal

### Location
- [gpt2_jepa.py:222-288](models/gpt2_jepa.py#L222-L288) - `create_reverse_sequence()` method

### Implementation
```python
def create_reverse_sequence(
    forward_input_ids: torch.Tensor,
    first_relation_token_id: int,  # ✅ Takes first relation token ID as parameter
) -> torch.Tensor:
```

### How it works
1. **Input**: `first_relation_token_id` (e.g., 16001 for `<r_0>`)
2. **Parity detection**:
   ```python
   first_is_odd = (first_relation_token_id % 2) == 1
   relation_is_odd = (relation % 2) == 1
   ```
3. **Inversion logic**:
   - If `<r_0>` is odd (16001): forward relations are odd, inverse are even
     - `16001 <r_0>` → `16002 <r_0_inv>` (add 1)
     - `16002 <r_0_inv>` → `16001 <r_0>` (subtract 1)
   - Pattern correctly handles all relation pairs

### Usage in training
```python
# train_jepa() at line 370-373
reverse_input_ids = model.create_reverse_sequence(
    forward_input_ids,
    first_relation_token_id  # ✅ Passed from dataloader
)
```

**Status**: ✅ **VERIFIED** - Uses `first_relation_token_id` to determine parity and correctly inverts relations

---

## ✅ Requirement 2: Selective Loss Based on Sample Type

### Location
- [gpt2_jepa.py:354-382](models/gpt2_jepa.py#L354-L382) - Training loop in `train_jepa()`
- [gpt2_jepa.py:150-160](models/gpt2_jepa.py#L150-L160) - Loss computation in `forward_jepa()`

### Implementation

#### Step 1: Extract type from batch (line 358-365)
```python
batch_types = batch['type']  # List of 'train' or 'atomic'

# Create boolean mask
compute_jepa = torch.tensor(
    [t == 'train' for t in batch_types],  # ✅ True only for 'train' type
    dtype=torch.bool,
    device=device
)
```

#### Step 2: Conditional reverse generation (line 367-373)
```python
reverse_input_ids = None
if compute_jepa.any():  # ✅ Only generate if at least one 'train' sample
    reverse_input_ids = model.create_reverse_sequence(...)
```

#### Step 3: Selective JEPA computation (line 150-160)
```python
if (self.jepa_config.use_jepa and
    reverse_input_ids is not None and
    compute_jepa is not None and
    compute_jepa.any()):  # ✅ Check if any samples need JEPA

    # Apply loss dropout
    if self.jepa_config.loss_dropout > 0:
        dropout_mask = torch.rand(batch_size, device=device) > self.jepa_config.loss_dropout
        compute_jepa = compute_jepa & dropout_mask  # ✅ Further filtering

    if compute_jepa.any():
        # Get indices where JEPA should be computed
        jepa_indices = compute_jepa.nonzero(as_tuple=True)[0]  # ✅ Only process 'train' samples
        forward_ids_jepa = forward_input_ids[jepa_indices]
        reverse_ids_jepa = reverse_input_ids[jepa_indices]
```

### Behavior by type

| Sample Type | NTP Loss | JEPA Loss | Reverse Generated |
|-------------|----------|-----------|-------------------|
| `"train"`   | ✅ Yes   | ✅ Yes    | ✅ Yes            |
| `"atomic"`  | ✅ Yes   | ❌ No     | ❌ No             |

**Status**: ✅ **VERIFIED** - Correctly applies both losses for 'train' type, only NTP for 'atomic'

---

## ✅ Requirement 3: Single Forward Pass with Block-Causal Masking

### Location
- [gpt2_jepa.py:23-69](models/gpt2_jepa.py#L23-L69) - `_create_block_causal_mask()`
- [gpt2_jepa.py:178-194](models/gpt2_jepa.py#L178-L194) - Combined forward pass

### Implementation

#### Step 1: Construct combined sequence (line 172-179)
```python
# Add k PRED tokens to forward sequence
if k > 0 and pred_token_id is not None:
    pred_tokens = torch.full((jepa_batch_size, k), pred_token_id, device=device)
    forward_with_pred = torch.cat([forward_ids_jepa, pred_tokens], dim=1)
else:
    forward_with_pred = forward_ids_jepa

# ✅ Concatenate into single sequence
combined_input_ids = torch.cat([forward_with_pred, reverse_ids_jepa], dim=1)
# Shape: (batch_size, seq_len + k + seq_len)
```

**Example with k=1**:
```
Forward:  [e1_f, e1_l, r, mask, e2_f, e2_l, PRED]  (7 tokens)
Reverse:  [e2_f, e2_l, r_inv, mask, e1_f, e1_l]     (6 tokens)
Combined: [e1_f, e1_l, r, mask, e2_f, e2_l, PRED, e2_f, e2_l, r_inv, mask, e1_f, e1_l]  (13 tokens)
```

#### Step 2: Create block-causal mask (line 182-187)
```python
attention_mask = self._create_block_causal_mask(
    seq_len=seq_len,      # Base sequence length (6)
    k_pred=k,             # Number of PRED tokens
    batch_size=jepa_batch_size,
    device=device,
)
```

#### Step 3: Block-causal mask structure (line 54-68)
```python
# Initialize all to -inf (no attention)
mask = torch.full((batch_size, 1, total_len, total_len), float('-inf'), device=device)

# Block 1: Forward + PRED tokens (causal within block)
forward_mask = torch.triu(..., diagonal=1)  # Upper triangle = -inf
mask[:, :, :forward_len, :forward_len] = forward_mask

# Block 2: Reverse sequence (causal within block)
reverse_mask = torch.triu(..., diagonal=1)
mask[:, :, forward_len:, forward_len:] = reverse_mask

# Cross-block attention remains -inf (forward and reverse DON'T see each other)
```

**Visualization** (k=1, seq_len=6):
```
Position:  0  1  2  3  4  5  6 | 7  8  9 10 11 12
Sequence: [e1_f e1_l r m e2_f e2_l P | e2_f e2_l ri m e1_f e1_l]
          └─────── Forward ──────┘   └───── Reverse ─────┘

Attention Mask (0 = can attend, -∞ = cannot attend):

         0  1  2  3  4  5  6 | 7  8  9 10 11 12
      ┌─────────────────────┬─────────────────────┐
    0 │ 0 -∞ -∞ -∞ -∞ -∞ -∞ │-∞ -∞ -∞ -∞ -∞ -∞    │ Forward
    1 │ 0  0 -∞ -∞ -∞ -∞ -∞ │-∞ -∞ -∞ -∞ -∞ -∞    │ block
    2 │ 0  0  0 -∞ -∞ -∞ -∞ │-∞ -∞ -∞ -∞ -∞ -∞    │ (causal)
    3 │ 0  0  0  0 -∞ -∞ -∞ │-∞ -∞ -∞ -∞ -∞ -∞    │
    4 │ 0  0  0  0  0 -∞ -∞ │-∞ -∞ -∞ -∞ -∞ -∞    │
    5 │ 0  0  0  0  0  0 -∞ │-∞ -∞ -∞ -∞ -∞ -∞    │
    6 │ 0  0  0  0  0  0  0 │-∞ -∞ -∞ -∞ -∞ -∞    │ ← PRED token
      ├─────────────────────┼─────────────────────┤
    7 │-∞ -∞ -∞ -∞ -∞ -∞ -∞ │ 0 -∞ -∞ -∞ -∞ -∞    │ Reverse
    8 │-∞ -∞ -∞ -∞ -∞ -∞ -∞ │ 0  0 -∞ -∞ -∞ -∞    │ block
    9 │-∞ -∞ -∞ -∞ -∞ -∞ -∞ │ 0  0  0 -∞ -∞ -∞    │ (causal)
   10 │-∞ -∞ -∞ -∞ -∞ -∞ -∞ │ 0  0  0  0 -∞ -∞    │
   11 │-∞ -∞ -∞ -∞ -∞ -∞ -∞ │ 0  0  0  0  0 -∞    │
   12 │-∞ -∞ -∞ -∞ -∞ -∞ -∞ │ 0  0  0  0  0  0    │ ← Last reverse token
      └─────────────────────┴─────────────────────┘
```

**Key properties**:
- ✅ Forward tokens attend causally to forward tokens
- ✅ Reverse tokens attend causally to reverse tokens
- ✅ **Forward and reverse do NOT attend to each other**
- ✅ PRED token sees all forward tokens (causal)
- ✅ Last reverse token sees all reverse tokens (causal)

#### Step 4: Single forward pass (line 190-194)
```python
# ✅ Single transformer call with custom mask
outputs = self.base_model.transformer(
    input_ids=combined_input_ids,
    attention_mask=attention_mask,  # Block-causal mask
)
hidden_states = outputs.last_hidden_state
```

#### Step 5: Extract embeddings (line 196-202)
```python
# Predicted embedding: last PRED token (or last forward token if k=0)
forward_end_idx = seq_len + k - 1
pred_emb = hidden_states[:, forward_end_idx, :]  # ✅ Position 6 (with k=1)

# Target embedding: last reverse token
target_emb = hidden_states[:, -1, :]  # ✅ Position 12 (with k=1)
```

**Status**: ✅ **VERIFIED** - Uses single forward pass with proper block-causal masking

---

## Summary

All three requirements are correctly implemented:

1. ✅ **Relation reversal** uses `first_relation_token_id` for accurate parity-based inversion
2. ✅ **Selective loss** applies JEPA+NTP for 'train' type, only NTP for 'atomic' type
3. ✅ **Single forward pass** with block-causal masking prevents forward/reverse cross-attention

## Training Flow Diagram

```
Batch Input
├── input_ids: [e1_f, e1_l, r, mask, e2_f, e2_l]
├── labels: [-100, -100, -100, -100, e2_f, e2_l]
└── type: 'train' or 'atomic'
    │
    ├─────────── NTP Loss (ALL samples) ─────────────┐
    │                                                 │
    │  forward_pass(input_ids, labels)                │
    │  └─> ntp_loss                                   │
    │                                                 ▼
    │                                           gamma * ntp_loss
    │                                                 │
    └─── If type == 'train': JEPA Loss ──────────────┤
         │                                            │
         ├─> create_reverse_sequence()                │
         │   (uses first_relation_token_id)          │
         │                                            │
         ├─> concat([forward + PRED, reverse])       │
         │                                            │
         ├─> block_causal_mask()                     │
         │   (forward and reverse isolated)          │
         │                                            │
         ├─> single_forward_pass()                   │
         │                                            │
         ├─> extract_embeddings()                    │
         │   ├─ pred_emb = hidden[6]                 │
         │   └─ target_emb = hidden[12]              │
         │                                            │
         └─> jepa_loss = distance(pred, target)      │
                                                      ▼
                                              lambda * jepa_loss
                                                      │
                                                      ▼
                                          total_loss = gamma*ntp + lambda*jepa
```

## Configuration

All parameters are correctly passed through the training pipeline:

```python
# From dataloader
special_tokens = get_special_token_ids(vocab_path)
pred_token_id = special_tokens['pred']                    # 16014
first_relation_token_id = special_tokens['first_relation']  # 16001

# To training function
train_jepa(
    model=model,
    train_dataloader=train_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    pred_token_id=pred_token_id,                          # ✅ Used for PRED tokens
    first_relation_token_id=first_relation_token_id,      # ✅ Used for relation inversion
    ...
)
```

**All requirements verified and ready for training!** 🚀
