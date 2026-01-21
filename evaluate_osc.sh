#!/bin/bash
#SBATCH --job-name=jepa_eval
#SBATCH --account=PAA0201
#SBATCH --time=00:30:00              # 30 minutes (typically finishes in 5-15 min)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=/fs/scratch/PAA0201/herb.45/logs/jepa_eval_%j.out
#SBATCH --error=/fs/scratch/PAA0201/herb.45/logs/jepa_eval_%j.err

cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source env/bin/activate

echo "=========================================="
echo "GPT2-JEPA EVALUATION (with MRR)"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "=========================================="
echo ""

# Set up environment
export CUDA_VISIBLE_DEVICES=0

# Print GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# ============================================
# CONFIGURE THESE PATHS
# ============================================
CHECKPOINT=/fs/scratch/PAA0201/herb.45/jepa_training/inversionidcomb10.50000.30000_k1_cosine_job3254906/checkpoints/best_model.pt
DATA_DIR=data/inversionidcomb10.50000.30000
OUTPUT_FILE=/fs/scratch/PAA0201/herb.45/jepa_training/inversionidcomb10.50000.30000_k1_cosine_job3254906/eval/eval_results.json
BATCH_SIZE=32
# ============================================

echo "Evaluation Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data directory: $DATA_DIR"
echo "  Output file: $OUTPUT_FILE"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT path in this script."
    exit 1
fi

# Create output directory if needed
mkdir -p $(dirname $OUTPUT_FILE)
mkdir -p /fs/scratch/PAA0201/herb.45/logs

# Set to true to compute per-layer MRR analysis (slower but more detailed)
PER_LAYER=true

# Run evaluation
echo "Starting evaluation on all splits (train, atomic, test)..."
echo "Computing: MRR (avg), MRR (token1), MRR (token2), Joint Accuracy"
if [ "$PER_LAYER" = true ]; then
    echo "Per-layer analysis: ENABLED (this will take longer)"
fi
echo ""

if [ "$PER_LAYER" = true ]; then
    python evaluate_all_splits.py \
        --checkpoint $CHECKPOINT \
        --data_dir $DATA_DIR \
        --batch_size $BATCH_SIZE \
        --output $OUTPUT_FILE \
        --per_layer
else
    python evaluate_all_splits.py \
        --checkpoint $CHECKPOINT \
        --data_dir $DATA_DIR \
        --batch_size $BATCH_SIZE \
        --output $OUTPUT_FILE
fi

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation finished at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
else
    echo "Evaluation exited with error code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
