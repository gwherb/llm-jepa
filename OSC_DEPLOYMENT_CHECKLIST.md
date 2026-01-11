# OSC Deployment Checklist

Use this checklist to ensure smooth deployment and training on Ohio Supercomputer Center.

---

## Pre-Deployment (On Local Machine)

### ✅ Files to Transfer to OSC

Make sure these files are uploaded to OSC:

**Core Model Files:**
- [ ] `models/gpt2_jepa.py`
- [ ] `models/gpt2_jepa_config.py`
- [ ] `models/gpt2_jepa_dataloader.py`

**Training Scripts:**
- [ ] `train_jepa.py`
- [ ] `train_jepa_osc.sh`
- [ ] `test_jepa_osc.sh`
- [ ] `check_training.sh`

**Data Files (should already be on OSC):**
- [ ] `data/inversionidcomb10.50000.30000/train.json`
- [ ] `data/inversionidcomb10.50000.30000/valid.json`
- [ ] `data/inversionidcomb10.50000.30000/vocab.json`

**Documentation (optional but recommended):**
- [ ] `TRAINING_VERIFICATION.md`
- [ ] `OSC_TRAINING_GUIDE.md`
- [ ] `IMPLEMENTATION_SUMMARY.md`

### Upload Command

```bash
# From your local machine
scp -r models/ train_jepa.py train_jepa_osc.sh test_jepa_osc.sh check_training.sh \
    herb.45@ascend.osc.edu:/fs/scratch/PAA0201/herb.45/llm-jepa/
```

---

## Initial Setup on OSC (One-Time)

### 1. Connect to OSC

```bash
ssh herb.45@ascend.osc.edu
```

### 2. Navigate to Project Directory

```bash
cd /fs/scratch/PAA0201/herb.45/llm-jepa
```

### 3. Verify Virtual Environment

```bash
# Activate environment
source env/bin/activate

# Verify Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

**Expected output:**
```
PyTorch: 2.x.x
Transformers: 4.x.x
```

**If packages missing:**
```bash
pip install torch transformers
```

### 4. Verify Data Files

```bash
ls -lh data/inversionidcomb10.50000.30000/
```

**Should see:**
```
train.json   (large file, ~100MB+)
valid.json   (medium file)
vocab.json   (small file)
```

**Test data loading:**
```bash
python -c "
import json
data = json.load(open('data/inversionidcomb10.50000.30000/train.json'))
print(f'Training samples: {len(data)}')
print(f'Sample: {data[0]}')
"
```

**Expected output:**
```
Training samples: 390000
Sample: {'input_text': [[...], [...], [...]], 'target_text': [[...]], 'type': 'train'}
```

### 5. Create Log Directory

```bash
mkdir -p /fs/scratch/PAA0201/herb.45/logs
```

### 6. Make Scripts Executable

```bash
chmod +x train_jepa_osc.sh test_jepa_osc.sh check_training.sh
```

---

## Quick Test (30 Minutes)

### 1. Submit Test Job

```bash
sbatch test_jepa_osc.sh
```

**Expected output:**
```
Submitted batch job XXXXXX
```

### 2. Check Job Status

```bash
squeue -u herb.45
```

**Possible states:**
- `PD` - Pending (waiting for resources)
- `R` - Running
- `CG` - Completing

### 3. Monitor Test Progress

```bash
# Watch job status
watch -n 5 'squeue -u herb.45'

# Or monitor output
tail -f /fs/scratch/PAA0201/herb.45/logs/jepa_test_XXXXXX.out
```

Replace `XXXXXX` with your actual job ID.

### 4. Verify Test Results

Once test completes (10-15 minutes):

```bash
# Check final output
tail -50 /fs/scratch/PAA0201/herb.45/logs/jepa_test_XXXXXX.out
```

**Look for:**
```
✓ TEST PASSED - All systems operational!
You can now run the full training with train_jepa_osc.sh
```

**If test fails:**
```bash
# Check error log
cat /fs/scratch/PAA0201/herb.45/logs/jepa_test_XXXXXX.err

# Common issues:
# - Out of memory: Reduce batch size in test_jepa_osc.sh
# - Import errors: Check virtual environment
# - Data errors: Verify data files exist and are valid JSON
```

---

## Full Training (24-48 Hours)

### Pre-Flight Checklist

Before submitting full training, verify:

- [ ] Quick test passed successfully
- [ ] No errors in test log
- [ ] Checkpoints saved correctly during test
- [ ] GPU was utilized during test
- [ ] Data loaded without errors

### 1. Review Training Configuration

Edit `train_jepa_osc.sh` if needed:

```bash
nano train_jepa_osc.sh
```

**Key parameters to review:**
```bash
N_LAYER=4              # Model layers (4 is good default)
BATCH_SIZE=256         # May need to reduce if OOM
LAMBDA_JEPA=1.0        # JEPA loss weight
K_PRED_TOK=1           # Predictor tokens (1 is good default)
```

### 2. Submit Full Training

```bash
sbatch train_jepa_osc.sh
```

**Save the job ID:**
```bash
Submitted batch job XXXXXX
```

### 3. Initial Verification (First 5 Minutes)

```bash
# Check job started
squeue -u herb.45

# Monitor initial output
tail -f /fs/scratch/PAA0201/herb.45/logs/jepa_train_XXXXXX.out
```

**Look for:**
- Job started message
- GPU information
- DataLoader loaded successfully
- Model initialized
- Training steps begin

**Press Ctrl+C to stop following log**

### 4. Ongoing Monitoring

Use the helper script:

```bash
./check_training.sh XXXXXX
```

Or manually check progress:

```bash
# Quick status
squeue -u herb.45

# Recent training steps
tail -50 /fs/scratch/PAA0201/herb.45/logs/jepa_train_XXXXXX.out | grep Step

# Check for errors
tail /fs/scratch/PAA0201/herb.45/logs/jepa_train_XXXXXX.err
```

---

## During Training

### Expected Timeline

**First Hour:**
- DataLoader loads data (~5 min)
- Model initializes (~1 min)
- Training begins
- First 100-500 steps complete
- First evaluation at step 5000

**Every 5000 steps:**
- Evaluation runs
- Validation loss logged

**Every 10000 steps:**
- Checkpoint saved

**After ~24-48 hours:**
- Training completes
- Final checkpoint saved

### Health Checks

Run these periodically (every few hours):

```bash
# 1. Job still running?
squeue -u herb.45

# 2. Training progressing?
./check_training.sh XXXXXX

# 3. Loss decreasing?
grep "Step" /fs/scratch/PAA0201/herb.45/logs/jepa_train_XXXXXX.out | tail -10

# 4. Any errors?
tail /fs/scratch/PAA0201/herb.45/logs/jepa_train_XXXXXX.err
```

### Normal Loss Values

**Expected loss progression:**
```
Initial:    Total ~8-10,  NTP ~7-9,   JEPA ~0.5-1.5
After 1K:   Total ~5-6,   NTP ~4-5,   JEPA ~0.4-0.8
After 10K:  Total ~3-4,   NTP ~2.5-3, JEPA ~0.3-0.5
After 50K:  Total ~2.5-3, NTP ~2-2.5, JEPA ~0.2-0.4
Final:      Total ~2-2.5, NTP ~1.5-2, JEPA ~0.15-0.3
```

**Warning signs:**
- Loss increasing
- Loss staying flat for >10K steps
- NaN or infinity values
- Very high gradient norms (>10)

---

## Troubleshooting

### Job Stuck in Queue (PD state)

```bash
# Check why job is pending
squeue -u herb.45 --start

# Possible reasons:
# - Cluster busy (wait)
# - Resource request too high (reduce in .sh file)
# - Account issue (contact OSC)
```

### Out of Memory (OOM)

**Symptoms:**
- Job killed with no error
- "CUDA out of memory" in error log

**Solutions:**
1. Reduce batch size in `train_jepa_osc.sh`:
   ```bash
   BATCH_SIZE=128  # or 64
   ```

2. Reduce model size:
   ```bash
   N_LAYER=2
   N_EMBD=512
   ```

3. Disable FP16 (remove `--fp16` flag)

### Training Stalled

**Symptoms:**
- No new output for >1 hour
- GPU utilization at 0%

**Actions:**
```bash
# Check if job is actually running
squeue -u herb.45

# Check node status
scontrol show job XXXXXX

# If truly stalled, cancel and restart
scancel XXXXXX
sbatch train_jepa_osc.sh
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
source env/bin/activate
pip install torch transformers
```

### Data Loading Errors

**Error:** `FileNotFoundError` or `JSONDecodeError`

**Solution:**
```bash
# Verify data files
ls -lh data/inversionidcomb10.50000.30000/

# Test JSON parsing
python -c "import json; json.load(open('data/inversionidcomb10.50000.30000/train.json'))"
```

---

## After Training Completes

### 1. Verify Completion

```bash
# Check final log output
tail -100 /fs/scratch/PAA0201/herb.45/logs/jepa_train_XXXXXX.out
```

**Look for:**
```
Training completed successfully!
Final model saved to ...
```

### 2. Locate Output Files

```bash
# Find output directory (shown in log)
OUTPUT_DIR=/fs/scratch/PAA0201/herb.45/jepa_training/...

# List checkpoints
ls -lh $OUTPUT_DIR/checkpoints/
```

**Should see:**
- Multiple checkpoint files (every 10K steps)
- `best_model.pt` (final model)
- `training_args.json`

### 3. Quick Model Check

```bash
python -c "
import torch
checkpoint = torch.load('$OUTPUT_DIR/checkpoints/best_model.pt', map_location='cpu')
print('Checkpoint keys:', checkpoint.keys())
print('Training completed at epoch:', checkpoint['epoch'])
print('Total steps:', checkpoint['step'])
"
```

### 4. Backup Important Files

```bash
# Copy to permanent storage
cp -r $OUTPUT_DIR ~/llm-jepa-results/run_$(date +%Y%m%d)/
```

---

## Quick Reference Commands

### Submit Jobs
```bash
sbatch test_jepa_osc.sh        # Quick test
sbatch train_jepa_osc.sh       # Full training
```

### Monitor Jobs
```bash
squeue -u herb.45              # List your jobs
./check_training.sh XXXXXX     # Detailed progress
tail -f logs/jepa_train_*.out  # Watch live
```

### Cancel Jobs
```bash
scancel XXXXXX                 # Cancel specific job
scancel -u herb.45             # Cancel all your jobs
```

### Check Resources
```bash
scontrol show job XXXXXX       # Detailed job info
sinfo                          # Cluster availability
```

---

## Success Criteria

### Test Job Success
- [ ] Job completes without errors
- [ ] "TEST PASSED" message appears
- [ ] At least one checkpoint saved
- [ ] No OOM errors
- [ ] GPU utilized during training

### Full Training Success
- [ ] Job runs for 24-48 hours
- [ ] Loss decreases steadily
- [ ] Checkpoints saved every 10K steps
- [ ] Final model saved
- [ ] No critical errors in log
- [ ] Training completes all epochs

---

## Next Steps After Training

1. **Evaluate Model:**
   - Test on held-out test set
   - Measure forward and reverse accuracy
   - Compare with baseline

2. **Analyze Results:**
   - Plot loss curves
   - Check reversal curse improvement
   - Analyze JEPA loss behavior

3. **Experiment:**
   - Try different k values
   - Test other distance metrics
   - Run ablation studies

---

## Emergency Contacts

**OSC Help Desk:**
- Email: oschelp@osc.edu
- Phone: (614) 292-9248
- Web: https://www.osc.edu/contact

**Documentation:**
- OSC Getting Started: https://www.osc.edu/resources/getting_started
- SLURM Guide: https://www.osc.edu/supercomputing/batch-processing-at-osc

---

**Ready to deploy!** Start with `sbatch test_jepa_osc.sh` ✅
