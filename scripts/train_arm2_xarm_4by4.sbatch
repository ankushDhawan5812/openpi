#!/bin/bash
#SBATCH --job-name=pi05_xarm_4by4
#SBATCH --account=arm
#SBATCH --partition=arm
#SBATCH --nodelist=arm2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a5000:8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Per-node scratch — /tmp is local to the node, so recreate the LeRobot
# dataset symlink here at job start.
export HF_HOME=/tmp/hf_cache
mkdir -p "${HF_HOME}/lerobot/ankushd"
ln -sfn /arm/u/ankushd/openpi-workspace/xarm_4by4_mug_machine_10hz \
        "${HF_HOME}/lerobot/ankushd/xarm_4by4_mug_machine_10hz"

# JAX / XLA tuning for 8 GPUs.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /arm/u/ankushd/openpi-workspace/openpi

# Make sure norm stats exist before launching the real run.
if [[ ! -f assets/pi05_xarm_4by4_lora_10hz/ankushd/xarm_4by4_mug_machine_10hz/norm_stats.json ]]; then
    echo "Computing norm stats..."
    uv run scripts/compute_norm_stats.py --config-name pi05_xarm_4by4_lora_10hz
fi

echo "Starting training on $(hostname), $(nvidia-smi -L | wc -l) GPUs visible."
nvidia-smi -L

uv run scripts/train.py pi05_xarm_4by4_lora_10hz \
    --exp-name=xarm_4by4_vla \
    --fsdp-devices=8
