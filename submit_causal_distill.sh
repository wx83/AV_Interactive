#!/usr/bin/env bash
#SBATCH --job-name=ltx2_causal_distill
#SBATCH --partition=sharedp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100:2
#SBATCH --requeue
#SBATCH --output=slurm-logs/causal_distill/ltx2_causal_distill.%N.%j.log
#SBATCH --error=slurm-logs/causal_distill/ltx2_causal_distill.%N.%j.log

set -euo pipefail

# Cleanup function to kill any lingering processes
cleanup() {
    echo "=== Cleaning up any lingering processes ==="
    pkill -f "train_causal_distill.py" || true
    sleep 2
}

# Set trap to cleanup on exit
trap cleanup EXIT
echo "=== Lightning multi-node job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) at $(date) ==="
echo "=== Using $SLURM_NNODES nodes, $SLURM_NTASKS total tasks ==="

# SLURM user tools setup (with error handling)
SLURM_USER_TOOL_ROOT=/usr/local/share/slurm-user/
if [ -d "$SLURM_USER_TOOL_ROOT" ]; then
  export PATH="$SLURM_USER_TOOL_ROOT:$PATH"
  set +u
  if ! source "$SLURM_USER_TOOL_ROOT/slurm-crusoe.bash_profile" 2>/dev/null; then
    echo "Warning: Could not source SLURM user tools profile"
    echo "   Continuing without SLURM user tools..."
  fi
  set -u
fi

# Activate conda environment
source ~/miniconda3/bin/activate ltx2

# Make sure srun-spawned processes also get the conda env
export PATH="$PATH"
export CONDA_DEFAULT_ENV=ltx2
export CONDA_PREFIX="$CONDA_PREFIX"

# Set up wandb environment variables
export WANDB_API_KEY="76cdc4261bf436617e661171fd41d80403e69e9b"
export WANDB_ENTITY="weihanx-university-of-michigan"
export WANDB_USERNAME="weihanx@umich.edu"
export WANDB_MODE="online"
export WANDB_SERVICE_WAIT="300"
export WANDB_INIT_TIMEOUT="300"
export WANDB_START_METHOD="thread"
export WANDB_CONSOLE="wrap"
echo "=== Wandb Configuration ==="
echo "WANDB_API_KEY: ${WANDB_API_KEY:0:10}...${WANDB_API_KEY: -4}"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "WANDB_MODE: $WANDB_MODE"
echo ""

# Set up environment
WORKDIR="/group2/ct/weihanx/LTX-2"
cd "$WORKDIR"

# Memory management: reduce fragmentation with expandable segments
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Multi-node setup for Lightning
# NOTE: Do NOT export LOCAL_RANK or RANK here — srun sets SLURM_PROCID and
# SLURM_LOCALID per-task. Setting them in the sbatch script bakes the same
# value into all tasks, breaking multi-GPU.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((29500 + ($SLURM_JOB_ID % 35500)))
export WORLD_SIZE=$SLURM_NTASKS

# ── Training parameters ──
CHECKPOINT_PATH="/group2/ct/weihanx/LTX-2/models/ltx-2-19b-distilled.safetensors"
GEMMA_ROOT="/group2/ct/weihanx/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized"
PROMPT_CSV="/group2/ct/weihanx/UltraVideo/short_with_potential_sounding_object_filtered.csv"
PROMPT_COLUMN="Summarized Description"
OUTPUT_DIR="checkpoints/causal_distill_19b"
PROMPT_POOL_SIZE=200

BATCH_SIZE=1
NUM_STEPS=5000
SAVE_EVERY=500
LR_GEN=1e-5
GRAD_CLIP=1.0
ACCUMULATE_GRAD_BATCHES=4

# Model / generation config
NUM_FRAMES=2
HEIGHT=384
WIDTH=640
NUM_FRAME_PER_BLOCK=2
LOCAL_ATTN_SIZE=4

# DMD mode: set to "--use-dmd" to enable 3-model DMD (needs ≥4× H100)
# Leave empty for 2-model velocity distillation (fits 2× H100)
USE_DMD=""
# USE_DMD="--use-dmd"
LR_CRITIC=1e-5
REAL_GUIDANCE_SCALE=4.5
FAKE_GUIDANCE_SCALE=0.0

# Distributed config — FSDP FULL_SHARD across 2 nodes × 2 GPUs = 4 GPUs
GPUS=2
NUM_NODES=$SLURM_NNODES
STRATEGY="fsdp"
PRECISION="bf16-true"

echo "=== Lightning multi-node setup ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT (calculated from job ID: $SLURM_JOB_ID)"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "Running on node $(hostname)"
echo ""
echo "=== Training Configuration ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Gemma root: $GEMMA_ROOT"
echo "Prompt CSV: $PROMPT_CSV"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Accumulate grad batches: $ACCUMULATE_GRAD_BATCHES"
echo "Effective batch size: $((BATCH_SIZE * GPUS * NUM_NODES * ACCUMULATE_GRAD_BATCHES))"
echo "Num steps: $NUM_STEPS"
echo "LR gen: $LR_GEN"
echo "Num frames: $NUM_FRAMES | Height: $HEIGHT | Width: $WIDTH"
echo "Strategy: $STRATEGY | Precision: $PRECISION"
echo "GPUs: $GPUS | Nodes: $NUM_NODES"
echo ""

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm-logs/causal_distill

echo "=== Checking master port availability ==="
lsof -ti:$MASTER_PORT | xargs --no-run-if-empty kill -9 || true
sleep 1
echo "Port $MASTER_PORT is ready for use"

# Resolve the conda python to use explicitly so srun inherits it
PYTHON_BIN=$(which python)
echo "=== Using python: $PYTHON_BIN ==="
echo "=== Python version: $($PYTHON_BIN --version) ==="

# ── Step 0: Prepare cache (single GPU, no srun) ──
# Encodes prompts and extracts pretrained weights once before multi-GPU training.
# Skips automatically if cache files already exist.
echo "=== Preparing training cache (single process) ==="
"$PYTHON_BIN" prepare_training_cache.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --gemma-root "$GEMMA_ROOT" \
    --prompt-csv "$PROMPT_CSV" \
    --prompt-column "$PROMPT_COLUMN" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-pool-size $PROMPT_POOL_SIZE \
    --seed 42

echo "=== Cache ready. Starting multi-GPU training ==="

# Launch PyTorch Lightning training
srun "$PYTHON_BIN" train_causal_distill.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --gemma-root "$GEMMA_ROOT" \
    --prompt-csv "$PROMPT_CSV" \
    --prompt-column "$PROMPT_COLUMN" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-pool-size $PROMPT_POOL_SIZE \
    --batch-size $BATCH_SIZE \
    --num-steps $NUM_STEPS \
    --save-every $SAVE_EVERY \
    --lr-gen $LR_GEN \
    --lr-critic $LR_CRITIC \
    --grad-clip $GRAD_CLIP \
    --accumulate-grad-batches $ACCUMULATE_GRAD_BATCHES \
    --num-frames $NUM_FRAMES \
    --height $HEIGHT \
    --width $WIDTH \
    --num-frame-per-block $NUM_FRAME_PER_BLOCK \
    --local-attn-size $LOCAL_ATTN_SIZE \
    $USE_DMD \
    --real-guidance-scale $REAL_GUIDANCE_SCALE \
    --fake-guidance-scale $FAKE_GUIDANCE_SCALE \
    --gpus $GPUS \
    --num-nodes $NUM_NODES \
    --strategy $STRATEGY \
    --precision $PRECISION \
    --gradient-checkpointing \
    --seed 42

echo "=== Job finished at $(date) ==="
