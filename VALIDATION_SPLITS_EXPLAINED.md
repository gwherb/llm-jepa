# Validation Splits Explained

## Data Structure

### Training Data (`train.json`)
- **390,000 samples total**
  - 300,000 with `type="train"` (both forward & reverse directions in training)
  - 90,000 with `type="atomic"` (only one direction in training)

### Validation Data (`valid.json`)
Contains **3 splits**, each with **3,000 samples**:

```python
{
  "train": [...],   # 3,000 samples from training 'train' type
  "atomic": [...],  # 3,000 samples from training 'atomic' type
  "test": [...]     # 3,000 HELD-OUT reverse directions (never seen!)
}
```

---

## The Three Splits Explained

### 1. **"train" Split** (Overfitting Check)

**What it is:**
- Random subsample of 3,000 samples from the 300k training samples with `type="train"`
- Both forward and reverse directions were in training data

**Purpose:**
- Monitor if model is overfitting vs. generalizing
- Standard ML validation set practice

**Example:**
```
Training: "Paris is-capital France" AND "France capital Paris"
Validation 'train': "Paris is-capital France" (seen both directions)
```

**What JEPA does here:**
- Computes JEPA loss (forward/reverse embeddings)
- Low JEPA loss = embeddings aligned well

**Expected behavior:**
- Loss should be similar to training loss
- If much higher → overfitting
- If much lower → underfitting

---

### 2. **"atomic" Split** (Partial Generalization)

**What it is:**
- Random subsample of 3,000 samples from the 90k atomic samples
- Only THIS direction was in training (reverse was NOT)
- But the reverse IS in the test set (held out)

**Purpose:**
- Test performance on samples where one direction was trained
- Intermediate difficulty between train and test

**Example:**
```
Training: "London is-capital UK" (only forward, reverse NOT in training)
Validation 'atomic': "London is-capital UK" (this direction was seen)
Validation 'test': "UK capital London" (held out - never seen!)
```

**What JEPA does here:**
- For samples with `type="atomic"`, JEPA loss is NOT computed
- Only NTP loss is measured

**Expected behavior:**
- Loss slightly higher than train (less memorization possible)
- Lower than test (this direction was actually trained)

---

### 3. **"test" Split** (🔑 THE KEY METRIC - Reversal Curse Test)

**What it is:**
- 3,000 samples of the **reverse directions** of atomic samples
- These reverse directions were **NEVER seen during training**
- This is the true reversal curse test!

**Purpose:**
- **Measure if model learned bidirectional reasoning**
- Test true reversal generalization
- THE metric that matters for the reversal curse

**Example:**
```
Training: "London is-capital UK" (only forward)
Validation 'test': "UK capital London" (NEVER SEEN - can model infer?)
```

**What JEPA should do:**
- Even though reverse wasn't seen, forward/reverse embeddings should align
- JEPA trains model to understand bidirectionality
- Should perform BETTER than baseline model

**Expected behavior:**
- **Without JEPA**: High loss (reversal curse - can't infer reverse)
- **With JEPA**: Lower loss (learned bidirectional structure)
- **Success metric**: Test loss similar to atomic loss (generalized!)

---

## Why This Design?

This 3-split validation design is standard in reversal curse research:

1. **Training samples**: Both directions to learn the relation
2. **Atomic samples**: One direction only
3. **Test samples**: The held-out reverse directions

**The hypothesis:**
- Baseline model: Sees "A→B" but can't infer "B→A" (reversal curse)
- JEPA model: Learns bidirectional embeddings, CAN infer "B→A"

---

## Evaluation Strategy

### During Training
Evaluate on **test split** periodically (every 5k steps):
```bash
--valid_split test
```

This monitors the KEY metric: reversal generalization.

### After Training
Run comprehensive evaluation on ALL splits:
```bash
python evaluate_all_splits.py \
  --checkpoint checkpoints/best_model.pt \
  --data_dir data/inversionidcomb10.50000.30000 \
  --batch_size 32
```

This provides complete picture:
- Train: Overfitting check
- Atomic: Performance on seen direction
- Test: **Reversal curse performance** (main result)

---

## Interpreting Results

### Good JEPA Performance

```
Split    | Loss  | Interpretation
---------|-------|------------------------------------------
train    | 2.5   | Well-trained, not overfitting
atomic   | 2.8   | Slight generalization gap (expected)
test     | 3.0   | ✓ GOOD - Can infer reverse directions!
```

**Success:** Test loss only slightly higher than atomic. JEPA helped!

### Poor Performance (Reversal Curse Persists)

```
Split    | Loss  | Interpretation
---------|-------|------------------------------------------
train    | 2.5   | Well-trained
atomic   | 2.8   | Good on seen directions
test     | 8.0   | ✗ BAD - Cannot infer reverse (reversal curse)
```

**Failure:** Large gap between atomic and test. JEPA didn't help enough.

### Overfitting

```
Split    | Loss  | Interpretation
---------|-------|------------------------------------------
train    | 2.0   | Low training loss
atomic   | 4.5   | Much higher - memorized training
test     | 5.0   | High - didn't generalize
```

**Problem:** Model memorized, didn't learn generalizable patterns.

---

## Updated Training Configuration

The training script now uses the **test split** for validation:

```bash
# train_jepa_osc.sh
--valid_split test  # Changed from 'train' to 'test'
```

This means during training, you'll see:
```
[Epoch 1/3] Step 5000 | Loss: 3.2145 | NTP: 2.8934 | JEPA: 0.3211
Eval Loss: 3.5421 | NTP: 3.5421 | JEPA: 0.0001  # <-- Test split loss
```

The eval loss is now the **reversal curse metric** - the most important number!

---

## Summary

| Split    | Size  | In Training? | Purpose                          | Key Metric For        |
|----------|-------|--------------|----------------------------------|-----------------------|
| train    | 3,000 | Yes (both)   | Overfitting check                | Training quality      |
| atomic   | 3,000 | Yes (1 dir)  | Seen direction performance       | Baseline              |
| **test** | 3,000 | **NO**       | **Reversal curse test**          | **MAIN RESULT**       |

**The test split is NOT redundant - it's the entire point of the experiment!**

It measures whether JEPA successfully teaches the model to perform bidirectional reasoning on never-before-seen reverse directions.
