#!/bin/bash
#SBATCH --job-name=jepa_test
#SBATCH --account=PAA0201
#SBATCH --time=00:30:00              # 30 minutes test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=/fs/scratch/PAA0201/herb.45/logs/jepa_test_%j.out
#SBATCH --error=/fs/scratch/PAA0201/herb.45/logs/jepa_test_%j.err

cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source env/bin/activate

echo "=========================================="
echo "GPT2-JEPA TEST RUN"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "=========================================="
echo ""

# Set up environment
export CUDA_VISIBLE_DEVICES=0

# Print GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Set parameters for SMALL test
DATASET=inversionidcomb10.50000.30000
N_LAYER=2                # Small model
N_EMBD=256               # Small embedding
N_HEAD=4                 # Few heads
BATCH_SIZE=16            # Small batch

# JEPA parameters
LAMBDA_JEPA=1.0
GAMMA_NTP=1.0
K_PRED_TOK=1
DISTANCE_METRIC=cosine

# Output directory
OUTPUT_DIR=/fs/scratch/PAA0201/herb.45/jepa_test/test_job${SLURM_JOB_ID}

echo "TEST Parameters (small model for quick verification):"
echo "  Dataset: $DATASET"
echo "  Model: ${N_LAYER} layers, ${N_EMBD} dim, ${N_HEAD} heads"
echo "  Batch size: $BATCH_SIZE"
echo "  JEPA: k=${K_PRED_TOK}, metric=${DISTANCE_METRIC}"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p /fs/scratch/PAA0201/herb.45/logs

# Run quick test (just 1 epoch with small model)
echo "Starting quick test..."
echo "This will verify:"
echo "  - DataLoader works"
echo "  - Model forward pass works"
echo "  - JEPA loss computation works"
echo "  - Training loop runs without errors"
echo ""

python train_jepa.py \
  --data_dir data/${DATASET}/ \
  --output_dir $OUTPUT_DIR \
  --n_layer $N_LAYER \
  --n_embd $N_EMBD \
  --n_head $N_HEAD \
  --n_positions 128 \
  --vocab_size 16015 \
  --batch_size $BATCH_SIZE \
  --eval_batch_size $BATCH_SIZE \
  --num_epochs 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --lambda_jepa $LAMBDA_JEPA \
  --gamma_ntp $GAMMA_NTP \
  --k_pred_tok $K_PRED_TOK \
  --distance_metric $DISTANCE_METRIC \
  --loss_dropout 0.0 \
  --use_jepa \
  --save_steps 1000 \
  --eval_steps 500 \
  --log_interval 50 \
  --num_workers 4 \
  --valid_split train \
  --fp16 \
  --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test finished at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TEST PASSED - All systems operational!"
    echo "You can now run the full training with train_jepa_osc.sh"
else
    echo "✗ TEST FAILED - Exit code: $EXIT_CODE"
    echo "Check the error log for details"
fi
echo "Output location: $OUTPUT_DIR"
echo "=========================================="

exit $EXIT_CODE
