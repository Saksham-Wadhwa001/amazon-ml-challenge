#!/bin/bash
# =============================================================================
# job_preprocess.sh
# -----------------------------------------------------------------------------
# Slurm batch script for the Amazon ML Business Entity Resolution preprocessing
# stage. Runs scripts/preprocess.py on an HPE DL380 Gen11 node (RHEL 9).
#
# Submit from the repository root:
#     mkdir -p logs           # ensure Slurm can create its .out/.err files
#     sbatch slurm_jobs/job_preprocess.sh
#
# Monitor live:
#     tail -f logs/preprocess_<jobid>.out
# =============================================================================

#SBATCH --job-name=preprocess_er
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# Exit on error and on failures inside a pipeline. (`-u` is intentionally left
# off so system profile scripts referencing unset vars don't abort the job.)
set -eo pipefail

# Run from the directory the job was submitted from (the repo root).
cd "${SLURM_SUBMIT_DIR:-.}"

echo "=================================================================="
echo "Job ID      : ${SLURM_JOB_ID:-n/a}"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "Submit dir  : ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "=================================================================="

# Slurm opens --output/--error before the script runs, so logs/ must already
# exist at submission time; this also covers any additional runtime logs.
mkdir -p logs

# --- Environment setup -------------------------------------------------------
source /etc/profile.d/anaconda.sh
source /etc/profile.d/cuda.sh
conda activate ml_challenge

# Pin Polars' thread pool to the allocated CPUs so we neither oversubscribe the
# node nor get throttled below our allocation.
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "Python      : $(which python)"
python --version
echo "Polars      : $(python -c 'import polars; print(polars.__version__)')"
echo "POLARS_MAX_THREADS=${POLARS_MAX_THREADS}"
# Preprocessing is CPU/IO-bound (Polars streaming engine); the H100 is requested
# per the cluster's standard GPU submission profile and for downstream stages.
nvidia-smi || echo "nvidia-smi unavailable (continuing; preprocessing is CPU-bound)."
echo "------------------------------------------------------------------"

# --- Run preprocessing -------------------------------------------------------
# `-u` forces unbuffered stdout/stderr so progress streams live to the log file.
python -u scripts/preprocess.py

echo "------------------------------------------------------------------"
echo "Finished    : $(date)"
