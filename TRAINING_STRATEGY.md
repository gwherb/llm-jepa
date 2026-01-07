# GPT-2 JEPA Training Strategy for Reversal Curse

## Data Structure

### Training Data (`train.json`)

Contains 390,000 examples split into two types:

1. **Type: "train"** (300,000 examples from GA entities)

   - Both forward and reverse directions exist in training set
   - Example pair:
     ```json
     {"input_text": [[e1_f, e1_l], [r], [mask]], "target_text": [[e2_f, e2_l]], "type": "train"}
     {"input_text": [[e2_f, e2_l], [r_inv], [mask]], "target_text": [[e1_f, e1_l]], "type": "train"}
     ```
   - **JEPA Eligible**: ✅ YES

2. **Type: "atomic"** (90,000 examples from GB entities)
   - Only ONE direction in training set
   - Reverse direction is held out in test set
   - Example:
     ```json
     {"input_text": [[e1_f, e1_l], [r], [mask]], "target_text": [[e2_f, e2_l]], "type": "atomic"}
     ```
   - **JEPA Eligible**: ❌ NO (would leak test data)

---

## Training Objectives

### 1. Next-Token Prediction (NTP) Loss

Applied to ALL examples (both "train" and "atomic"):

**Input**: `[e1_first, e1_last, relation]`
**Target**: `[e2_first, e2_last]`

**Sequence Format** (NO predictor tokens):

```
Input IDs:  [e1_f, e1_l, r, e2_f, e2_l]
Labels:     [-100, -100, -100, e2_f, e2_l]
```

Only target entity tokens are supervised (rest masked with -100).

**Key Point**: NTP loss does NOT use predictor tokens - only the actual relation tokens.

### 2. JEPA Loss (Selective Application)

Applied ONLY to examples with `type="train"`:

**Forward sequence** (WITH predictor tokens): `[e1_f, e1_l, r, e2_f, e2_l, PRED_0, PRED_1, PRED_2, PRED_3]`
**Reverse sequence**: `[e2_f, e2_l, r_inv, e1_f, e1_l]`

**Loss Computation**:

1. Extract embedding from last predictor token of forward sequence
2. Extract embedding from last token of reverse sequence
3. Compute cosine similarity between embeddings
4. JEPA loss = 1 - cosine_similarity
5. **Filter**: Only compute for examples where `compute_jepa=True`

**Training Logic**:

- `type="train"` → Compute BOTH NTP loss AND JEPA loss
- `type="atomic"` → Compute ONLY NTP loss (skip JEPA)

**Total Loss**: `gamma * NTP_loss + lbd * JEPA_loss`
