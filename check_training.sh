#!/bin/bash
# Helper script to check training progress on OSC
# Usage: ./check_training.sh [job_id]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "GPT2-JEPA Training Status Checker"
echo "=========================================="
echo ""

# Get job ID from argument or find latest
if [ -z "$1" ]; then
    echo "Looking for latest JEPA training job..."
    LATEST_JOB=$(squeue -u herb.45 -n jepa_train -h -o "%i" | head -1)
    if [ -z "$LATEST_JOB" ]; then
        echo -e "${YELLOW}No running JEPA training jobs found.${NC}"
        echo "Checking recent jobs..."
        sacct -u herb.45 -n jepa_train --format=JobID,JobName,State,Elapsed,Start -P | head -5
        exit 0
    fi
    JOB_ID=$LATEST_JOB
else
    JOB_ID=$1
fi

echo "Checking job: $JOB_ID"
echo ""

# Check if job is running
JOB_STATE=$(scontrol show job $JOB_ID 2>/dev/null | grep "JobState" | awk '{print $1}' | cut -d'=' -f2)

if [ -z "$JOB_STATE" ]; then
    echo -e "${RED}Job $JOB_ID not found.${NC}"
    echo "Recent jobs:"
    sacct -u herb.45 --format=JobID,JobName,State,Elapsed,Start -P | grep jepa | head -5
    exit 1
fi

echo -e "Job State: ${GREEN}$JOB_STATE${NC}"

# Get job info
echo ""
echo "Job Information:"
scontrol show job $JOB_ID | grep -E "JobName|RunTime|TimeLimit|NodeList|WorkDir"

# Find log files
LOG_OUT="/fs/scratch/PAA0201/herb.45/logs/jepa_train_${JOB_ID}.out"
LOG_ERR="/fs/scratch/PAA0201/herb.45/logs/jepa_train_${JOB_ID}.err"

echo ""
echo "Log files:"
echo "  Output: $LOG_OUT"
echo "  Error:  $LOG_ERR"

# Check if log files exist
if [ ! -f "$LOG_OUT" ]; then
    echo -e "${YELLOW}Output log not found yet (job may be queued)${NC}"
    exit 0
fi

echo ""
echo "=========================================="
echo "Training Progress"
echo "=========================================="

# Get last 20 lines of output showing progress
echo ""
echo "Recent training steps:"
tail -20 "$LOG_OUT" | grep -E "(Step|Epoch|Loss|Evaluation|Checkpoint)" || echo "No training output yet..."

echo ""
echo "=========================================="
echo "Loss Summary"
echo "=========================================="

# Extract and show loss progression
echo ""
echo "Loss progression (last 10 logged steps):"
tail -100 "$LOG_OUT" | grep "Step" | tail -10 | awk '{
    for (i=1; i<=NF; i++) {
        if ($i ~ /^Step/) step=$(i+1)
        if ($i ~ /^Loss=/) loss=$(i+1)
        if ($i ~ /^NTP=/) ntp=$(i+1)
        if ($i ~ /^JEPA=/) jepa=$(i+1)
    }
    printf "  Step %s: Total=%.4f, NTP=%.4f, JEPA=%.4f\n", step, loss, ntp, jepa
}'

echo ""
echo "=========================================="
echo "Checkpoints"
echo "=========================================="

# Find output directory
OUTPUT_DIR=$(grep "Output directory:" "$LOG_OUT" | tail -1 | awk '{print $NF}')

if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR/checkpoints" ]; then
    echo ""
    echo "Saved checkpoints in $OUTPUT_DIR/checkpoints:"
    ls -lh "$OUTPUT_DIR/checkpoints" | tail -5
    echo ""
    CHECKPOINT_COUNT=$(ls "$OUTPUT_DIR/checkpoints"/*.pt 2>/dev/null | wc -l)
    echo "Total checkpoints: $CHECKPOINT_COUNT"
else
    echo "No checkpoints found yet."
fi

echo ""
echo "=========================================="
echo "Errors (if any)"
echo "=========================================="

if [ -f "$LOG_ERR" ]; then
    ERR_SIZE=$(stat -f%z "$LOG_ERR" 2>/dev/null || stat -c%s "$LOG_ERR" 2>/dev/null)
    if [ "$ERR_SIZE" -gt 0 ]; then
        echo -e "${RED}Error log has content!${NC}"
        echo "Last 20 lines of error log:"
        tail -20 "$LOG_ERR"
    else
        echo -e "${GREEN}No errors logged.${NC}"
    fi
else
    echo "Error log not found yet."
fi

echo ""
echo "=========================================="
echo "GPU Usage (if running)"
echo "=========================================="

NODE=$(scontrol show job $JOB_ID 2>/dev/null | grep "NodeList" | awk '{print $1}' | cut -d'=' -f2)

if [ "$JOB_STATE" == "RUNNING" ] && [ -n "$NODE" ]; then
    echo ""
    echo "GPU status on $NODE:"
    ssh $NODE "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits" 2>/dev/null || echo "Could not connect to node"
else
    echo "Job not currently running on a node."
fi

echo ""
echo "=========================================="
echo "Quick Commands"
echo "=========================================="
echo ""
echo "To watch training in real-time:"
echo "  tail -f $LOG_OUT"
echo ""
echo "To cancel this job:"
echo "  scancel $JOB_ID"
echo ""
echo "To check all your jobs:"
echo "  squeue -u herb.45"
echo ""
echo "=========================================="
