#!/bin/bash
# =============================================================================
# job_entity_resolution.sh
# -----------------------------------------------------------------------------
# Slurm batch script for the Entity Resolution matching stage on the HPE DL380
# Gen11 H100 node (RHEL 9). Runs scripts/entity_resolution.py end-to-end:
# blocking -> features (Qwen3-Embedding + Qwen3-Reranker) -> LightGBM -> outputs.
#
# Submit from the repo root:
#     mkdir -p logs
#     sbatch slurm_jobs/job_entity_resolution.sh
#
# Monitor:  tail -f logs/er_<jobid>.out
# =============================================================================

#SBATCH --job-name=er_match
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/er_%j.out
#SBATCH --error=logs/er_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "=================================================================="
echo "Job ID    : ${SLURM_JOB_ID:-n/a}   Node: $(hostname)   $(date)"
echo "=================================================================="
mkdir -p logs output

# --- Environment -------------------------------------------------------------
source /etc/profile.d/anaconda.sh
source /etc/profile.d/cuda.sh
conda activate ml_challenge

export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
# Keep model downloads inside the project (first run pulls the Qwen3 weights).
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

echo "Python : $(which python)"; python --version
nvidia-smi || echo "nvidia-smi unavailable."
echo "------------------------------------------------------------------"

# --- Run ---------------------------------------------------------------------
# Best-accuracy config (Apache-2.0, <= 8B). Drop to Qwen3-*-4B for more speed,
# or add --no-reranker for a fast first pass.
python -u scripts/entity_resolution.py \
    --processed-dir data/processed \
    --ground-truth data/train_ground_truth.tsv \
    --output-dir output \
    --emb-model Qwen/Qwen3-Embedding-8B \
    --rerank-model Qwen/Qwen3-Reranker-8B \
    --k-tfidf 50 --k-emb 50

echo "------------------------------------------------------------------"
echo "Finished : $(date)"
