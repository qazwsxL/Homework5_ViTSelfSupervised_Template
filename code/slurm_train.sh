#!/bin/bash
#SBATCH -J hw5_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# HW5: Vision Transformers and Self-Supervised Learning
# Run all five tasks sequentially.

echo "Starting HW5 training on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd "$(dirname "$0")"

echo "=== Task 0: Attention Visualization ==="
uv run python main.py --task t0_attention

echo "=== Task 1: End-to-End ViT ==="
uv run python main.py --task t1_endtoend

echo "=== Task 2: Rotation Prediction ==="
uv run python main.py --task t2_rotation

echo "=== Task 3: Mini-DINO Pretraining ==="
uv run python main.py --task t3_dino

echo "=== Task 4: Transfer Evaluation ==="
uv run python main.py --task t4_transfer

echo "All tasks complete at $(date)"
