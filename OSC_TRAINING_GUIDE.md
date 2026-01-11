# OSC Training Guide for GPT2-JEPA

This guide explains how to train the GPT2-JEPA model on the Ohio Supercomputer Center.

## Files Overview

### Training Scripts
- **`train_jepa.py`**: Main Python training script
- **`train_jepa_osc.sh`**: SLURM job script for full training (72 hours)
- **`test_jepa_osc.sh`**: SLURM job script for quick test (30 minutes)
- **`test_training_local.py`**: Local CPU test script (for Windows/Mac)

### Model Files
- **`models/gpt2_jepa.py`**: GPT2-JEPA model implementation
- **`models/gpt2_jepa_config.py`**: JEPA configuration
- **`models/gpt2_jepa_dataloader.py`**: DataLoader for reversal curse data

### Documentation
- **`TRAINING_VERIFICATION.md`**: Detailed verification of training requirements
- **`TRAINING_STRATEGY.md`**: Original training strategy document
- **`README_DATALOADER.md`**: DataLoader documentation

## Quick Start on OSC

### 1. Setup (One-time)

```bash
# SSH to OSC
ssh herb.45@ascend.osc.edu

# Navigate to project directory
cd /fs/scratch/PAA0201/herb.45/llm-jepa

# Activate virtual environment (should already exist from previous work)
source env/bin/activate

# Verify PyTorch is installed
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 2. Quick Test (Recommended First)

Run a quick 30-minute test to verify everything works:

```bash
# Submit test job
sbatch test_jepa_osc.sh

# Check job status
squeue -u herb.45

# Monitor output (replace JOBID with actual job ID)
tail -f /fs/scratch/PAA0201/herb.45/logs/jepa_test_JOBID.out
```

**What the test does:**
- Loads a small 2-layer model (256 dim, 4 heads)
- Runs 1 epoch with batch size 16
- Verifies DataLoader, forward pass, JEPA loss, and training loop
- Takes ~10-15 minutes on A100

**If test passes:** Proceed to full training
**If test fails:** Check error log at `/fs/scratch/PAA0201/herb.45/logs/jepa_test_JOBID.err`

### 3. Full Training

Once the test passes, submit the full training job:

```bash
# Submit full training job
sbatch train_jepa_osc.sh

# Check job status
squeue -u herb.45

# Monitor training progress
tail -f /fs/scratch/PAA0201/herb.45/logs/jepa_train_JOBID.out

# Check for errors
tail -f /fs/scratch/PAA0201/herb.45/logs/jepa_train_JOBID.err
```

**Full training configuration:**
- Model: 4 layers, 768 dim, 12 heads
- Batch size: 256
- Epochs: 3
- JEPA: k=1 predictor token, cosine distance
- Expected runtime: 24-48 hours
- Saves checkpoints every 10,000 steps

## Configuration Options

### Model Size

Edit `train_jepa_osc.sh` to adjust model size:

```bash
N_LAYER=4        # Number of transformer layers (2, 4, 6, 12)
N_EMBD=768       # Embedding dimension (256, 512, 768, 1024)
N_HEAD=12        # Attention heads (must divide N_EMBD)
```

### JEPA Parameters

```bash
LAMBDA_JEPA=1.0      # JEPA loss weight (0.0 to disable)
GAMMA_NTP=1.0        # NTP loss weight
K_PRED_TOK=1         # Number of predictor tokens (0, 1, 2, 3, 4)
DISTANCE_METRIC=cosine  # cosine, l2, or mse
LOSS_DROPOUT=0.0     # Dropout percentage (0.0 to 1.0)
```

### Training Hyperparameters

```bash
BATCH_SIZE=256       # Training batch size
LEARNING_RATE=1e-4   # Learning rate
WEIGHT_DECAY=0.01    # Weight decay for AdamW
```

## Output Structure

Training outputs are saved to `/fs/scratch/PAA0201/herb.45/jepa_training/`:

```
jepa_training/
└── inversionidcomb10.50000.30000_k1_cosine_jobXXXXXX/
    ├── training_args.json           # Training configuration
    ├── training.log                 # Training logs
    └── checkpoints/
        ├── checkpoint_epoch0_step10000.pt
        ├── checkpoint_epoch1_step20000.pt
        └── best_model.pt            # Final model
```

## Monitoring Training

### Check Job Status

```bash
# List your jobs
squeue -u herb.45

# Get detailed job info
scontrol show job JOBID
```

### Monitor Progress

```bash
# Watch training output in real-time
tail -f /fs/scratch/PAA0201/herb.45/logs/jepa_train_JOBID.out

# Check for errors
tail -f /fs/scratch/PAA0201/herb.45/logs/jepa_train_JOBID.err

# Check GPU usage (if job is running)
ssh nodeXXX  # Replace with actual node from squeue
nvidia-smi
```

### Training Logs

The training script logs:
- Step number and epoch
- Total loss, NTP loss, JEPA loss
- Gradient norm
- Evaluation metrics every 5,000 steps
- Checkpoint saves every 10,000 steps

Example output:
```
Epoch 1/3, Step 100: Loss=3.2145, NTP=2.8934, JEPA=0.3211, GradNorm=0.8234
Epoch 1/3, Step 200: Loss=3.0821, NTP=2.7456, JEPA=0.3365, GradNorm=0.7891
...
Evaluation at step 5000: Val Loss=2.8765, Val NTP=2.5432, Val JEPA=0.3333
Checkpoint saved at step 10000
```

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA out of memory errors:

1. **Reduce batch size:**
   ```bash
   BATCH_SIZE=128  # or 64, 32
   ```

2. **Use gradient accumulation:**
   Add to `train_jepa.py`:
   ```python
   --gradient_accumulation_steps 2
   ```

3. **Reduce model size:**
   ```bash
   N_LAYER=2
   N_EMBD=512
   ```

### Job Killed (Time Limit)

If job exceeds 72 hours:

1. **Resume from checkpoint:**
   ```bash
   python train_jepa.py \
     --resume_from /path/to/checkpoint.pt \
     [other args...]
   ```

2. **Increase time limit** (edit SBATCH header):
   ```bash
   #SBATCH --time=96:00:00  # 4 days
   ```

### DataLoader Errors

If you get data loading errors:

1. **Check data paths:**
   ```bash
   ls data/inversionidcomb10.50000.30000/
   # Should see: train.json, valid.json, vocab.json
   ```

2. **Verify data format:**
   ```bash
   python -c "import json; data = json.load(open('data/inversionidcomb10.50000.30000/train.json')); print(type(data), len(data))"
   # Should print: <class 'list'> 390000
   ```

### Import Errors

If you get module import errors:

```bash
# Make sure you're in the project directory
cd /fs/scratch/PAA0201/herb.45/llm-jepa

# Activate environment
source env/bin/activate

# Install any missing dependencies
pip install transformers torch
```

## Comparing with Baseline

To compare JEPA training with baseline GPT2:

1. **Run baseline** (your existing `comb2_full.sh`)
2. **Run JEPA** (`train_jepa_osc.sh`)
3. **Compare outputs:**
   - Training loss curves
   - Validation performance
   - Reversal accuracy (forward vs reverse)

## Advanced Options

### Multiple k Values

To test different numbers of predictor tokens, edit:

```bash
K_PRED_TOK=2  # Try k=0,1,2,3,4
```

### Different Distance Metrics

To test different JEPA distance metrics:

```bash
DISTANCE_METRIC=l2    # or mse, cosine
```

### Ablation: Disable JEPA

To run baseline without JEPA (only NTP):

```bash
LAMBDA_JEPA=0.0  # Set JEPA weight to 0
```

Or remove `--use_jepa` flag from `train_jepa.py` call.

## Expected Results

After training completes, you should see:

1. **Checkpoints** saved every 10,000 steps
2. **Final model** at `best_model.pt`
3. **Training logs** with loss curves
4. **Validation metrics** showing:
   - NTP loss (should decrease)
   - JEPA loss (should decrease)
   - Combined loss (should decrease)

Typical loss values:
- Initial: 8-10
- After epoch 1: 3-4
- After epoch 3: 2-3

## Next Steps

After training:

1. **Evaluate on test set:**
   ```python
   from models.gpt2_jepa import evaluate_jepa
   # Load model and run evaluation
   ```

2. **Test reversal accuracy:**
   - Forward: "A is B" → predict B
   - Reverse: "B is A" → predict A
   - Compare accuracy

3. **Compare with baseline:**
   - Load baseline model from `comb2_full.sh` output
   - Compare reversal performance

## Contact

For OSC-specific issues:
- OSC Help: oschelp@osc.edu
- Documentation: https://www.osc.edu/resources/getting_started

For code issues:
- Check TRAINING_VERIFICATION.md
- Review error logs in `/fs/scratch/PAA0201/herb.45/logs/`
