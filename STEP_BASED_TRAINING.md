# Step-Based Training Implementation

## Summary

The training script has been updated to support **step-based training** instead of epoch-based training, matching the baseline job configuration. This allows for more precise control over training duration and checkpoint saving.

## Key Changes

### 1. Training Arguments (train_jepa.py)

Added new command-line arguments:

```bash
--max_steps 3000000              # Maximum training steps (overrides num_epochs)
--save_step_dense 30000          # Save more frequently until this step
--save_step_dense_interval 5000  # Interval for dense checkpoint saving
```

Updated existing arguments:
```bash
--save_steps 50000              # Regular checkpoint interval (after dense period)
--eval_steps 5000               # Evaluation interval
```

### 2. Training Loop (models/gpt2_jepa.py)

**Modified `train_jepa()` function to support:**

- **Dual training modes**: Step-based (max_steps) or epoch-based (num_epochs)
- **Early stopping**: Terminates when `max_steps` is reached
- **Step-based evaluation**: Evaluates every `eval_steps` instead of per-epoch
- **Dense checkpoint saving**: Saves more frequently early in training
  - Every 5,000 steps until step 30,000
  - Every 50,000 steps after step 30,000

**Checkpoint naming**: Changed from `checkpoint_epoch_X.pt` to `checkpoint_step_X.pt`

### 3. Updated Training Script (train_jepa_osc.sh)

```bash
# OLD (epoch-based)
--num_epochs 3 \
--save_steps 10000 \
--eval_steps 5000 \

# NEW (step-based)
--max_steps 3000000 \
--save_steps 50000 \
--save_step_dense 30000 \
--save_step_dense_interval 5000 \
--eval_steps 5000 \
```

## Training Configuration

### Current Setup (matching baseline job)

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_steps | 3,000,000 | Total training steps |
| save_steps | 50,000 | Regular checkpoint interval |
| save_step_dense | 30,000 | Dense saving until this step |
| save_step_dense_interval | 5,000 | Dense checkpoint interval |
| eval_steps | 5,000 | Evaluation frequency |
| batch_size | 256 | Samples per batch |
| dataset_size | 390,000 | Training samples |

### Training Duration

- **Steps per epoch**: 390,000 / 256 = ~1,523 steps
- **Total epochs**: 3,000,000 / 1,523 = ~1,970 epochs
- **Estimated runtime**: 48-72 hours on A100 GPU

### Checkpoint Schedule

**Dense saving (steps 1-30,000):**
- Checkpoints at: 5k, 10k, 15k, 20k, 25k, 30k (6 checkpoints)

**Regular saving (steps 30,001-3,000,000):**
- Checkpoints at: 50k, 100k, 150k, ..., 3,000k (60 checkpoints)

**Total checkpoints**: ~66 checkpoints

## Usage

### Run Full Training

```bash
sbatch train_jepa_osc.sh
```

This will train for 3 million steps with the configuration above.

### Run Quick Test

```bash
sbatch test_jepa_osc.sh
```

This runs 1 epoch (~1,523 steps) to verify everything works.

### Monitor Training

```bash
# Check job status
squeue -u $USER

# View live output
tail -f /fs/scratch/PAA0201/herb.45/logs/jepa_train_JOBID.out

# Check checkpoints
ls -lh /fs/scratch/PAA0201/herb.45/jepa_training/*/checkpoints/
```

## Evaluation

### During Training

The model evaluates on the **test split** every 5,000 steps. This provides:
- Real-time monitoring of reversal curse performance
- Early detection of training issues
- Ability to identify best checkpoint

### After Training

Run comprehensive evaluation on all splits:

```bash
python evaluate_all_splits.py \
  --checkpoint /path/to/checkpoint_step_3000000.pt \
  --data_dir data/inversionidcomb10.50000.30000 \
  --batch_size 32 \
  --output results.json
```

This evaluates on:
- **train split**: Overfitting check
- **atomic split**: Seen direction performance
- **test split**: Reversal curse test (main metric)

## Resuming Training

If training is interrupted, resume from a checkpoint:

```bash
python train_jepa.py \
  --resume_from /path/to/checkpoint_step_X.pt \
  --max_steps 3000000 \
  [... other args ...]
```

The training will continue from step X until max_steps.

## Comparison: Old vs New

| Aspect | Old (Epoch-based) | New (Step-based) |
|--------|-------------------|------------------|
| Training duration | 3 epochs (~4,570 steps) | 3M steps (~1,970 epochs) |
| Checkpoints | Per epoch (3 total) | Per step schedule (66 total) |
| Evaluation | Per epoch | Every 5k steps |
| Total runtime | ~15 minutes | ~48-72 hours |
| Matches baseline | ❌ No | ✅ Yes |

## Benefits

1. **Precise control**: Train for exact number of steps, not approximate epochs
2. **Better monitoring**: Evaluate every 5k steps instead of waiting for epoch end
3. **Dense early checkpoints**: More frequent saves during critical early training
4. **Baseline consistency**: Matches the proven baseline job configuration
5. **Resume flexibility**: Can resume from any checkpoint and continue to max_steps

## Notes

- The test script (`test_jepa_osc.sh`) still uses `--num_epochs 1` for quick verification
- Both training modes are supported: specify either `--max_steps` OR `--num_epochs`
- If `--max_steps` is specified, it takes precedence over `--num_epochs`
- Checkpoint filenames changed from `checkpoint_epoch_X.pt` to `checkpoint_step_X.pt`
