#!/bin/bash
#SBATCH --job-name=jepa_train
#SBATCH --account=PAA0201
#SBATCH --time=72:00:00              # 72 hours (3 days)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # More CPUs for data loading
#SBATCH --gpus-per-node=1            # 1 GPU on Ascend (A100)
#SBATCH --mem=128GB                  # Large CPU memory
#SBATCH --output=/fs/scratch/PAA0201/herb.45/logs/jepa_train_%j.out
#SBATCH --error=/fs/scratch/PAA0201/herb.45/logs/jepa_train_%j.err

cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source env/bin/activate

echo "=========================================="
echo "GPT2-JEPA TRAINING"
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

# Set parameters
DATASET=inversionidcomb10.50000.30000
N_LAYER=4
N_EMBD=768
N_HEAD=12
BATCH_SIZE=256
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01

# JEPA parameters
LAMBDA_JEPA=1.0
GAMMA_NTP=1.0
K_PRED_TOK=1
LOSS_DROPOUT=0.0
DISTANCE_METRIC=cosine

# Output directory in scratch with unique job ID
OUTPUT_DIR=/fs/scratch/PAA0201/herb.45/jepa_training/${DATASET}_k${K_PRED_TOK}_${DISTANCE_METRIC}_job${SLURM_JOB_ID}

echo "Training Parameters:"
echo "  Dataset: $DATASET"
echo "  Model: ${N_LAYER} layers, ${N_EMBD} dim, ${N_HEAD} heads"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo ""
echo "JEPA Parameters:"
echo "  Lambda (JEPA weight): $LAMBDA_JEPA"
echo "  Gamma (NTP weight): $GAMMA_NTP"
echo "  K (predictor tokens): $K_PRED_TOK"
echo "  Distance metric: $DISTANCE_METRIC"
echo "  Loss dropout: $LOSS_DROPOUT"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory and logs directory
mkdir -p $OUTPUT_DIR
mkdir -p /fs/scratch/PAA0201/herb.45/logs

# Run the training
echo "Starting GPT2-JEPA training..."
echo "Expected runtime: 24-48 hours"
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
  --num_epochs 3 \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --lambda_jepa $LAMBDA_JEPA \
  --gamma_ntp $GAMMA_NTP \
  --k_pred_tok $K_PRED_TOK \
  --distance_metric $DISTANCE_METRIC \
  --loss_dropout $LOSS_DROPOUT \
  --use_jepa \
  --save_steps 10000 \
  --eval_steps 5000 \
  --log_interval 100 \
  --num_workers 8 \
  --valid_split test \
  --fp16 \
  --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training exited with error code: $EXIT_CODE"
fi
echo "Output location: $OUTPUT_DIR"
echo "=========================================="

exit $EXIT_CODE
