#!/bin/bash
#SBATCH -J vdm_test_ep5
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm/%x.%j.err
#SBATCH -p a40
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch

# ---- Paths ----
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_VDM"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

CKPT_DIR="${PROJECT_ROOT}/results/vdm/${RUN_ID}"
LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
SLURM_DIR="${PROJECT_ROOT}/logs/slurm"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${SLURM_DIR}"

# ---- Env ----
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_MODULE_LOADING=LAZY
export HDF5_USE_FILE_LOCKING=FALSE
export GAL2DM_LOGDIR="${LOG_DIR}"
export NCCL_P2P_DISABLE=1
ulimit -n 65535

cd "${PROJECT_ROOT}"

echo "=== [JOB STARTED] $(date) on $(hostname) ==="
START_TIME=$(date +%s)

which python

python - <<'PY'
import torch, platform, os
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA (Torch):", getattr(torch.version, "cuda", None))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name[0]:", torch.cuda.get_device_name(0))
PY

nvidia-smi || echo "nvidia-smi not available"

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.stderr.write("[FATAL] CUDA not available.\n")
    sys.exit(2)
PY

# ---- Training ----
EPOCHS=30
SAMPLING=0.1
MODEL_NAME="vdm_cunet3d_test"
CONSOLE_LOG="${LOG_DIR}/${MODEL_NAME}_ep${EPOCHS}.console.log"
touch "${CONSOLE_LOG}"

echo "Launching VDM training..."
srun --ntasks=1 python -u -m src.train \
  --yaml_path "${YAML_PATH}" \
  --target_field rho \
  --train_val_split 0.8 \
  --sample_fraction ${SAMPLING} \
  --batch_size 1 \
  --num_workers 6 \
  --epochs ${EPOCHS} \
  --min_lr 1e-4 \
  --max_lr 1e-3 \
  --cycle_length 8 \
  --ckpt_dir "${CKPT_DIR}" \
  --seed 42 \
  --device cuda \
  --keep_two_channels \
  --amp \
  --validate_keys False \
  --exclude_list "${PROJECT_ROOT}/etc/filelists/exclude_bad_all.txt" \
  2>&1 | tee -a "${CONSOLE_LOG}"

# ====== ⏱ 총 소요 시간 계산 ======
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

H=$((ELAPSED / 3600))
M=$(( (ELAPSED % 3600) / 60 ))
S=$((ELAPSED % 60))

echo "--------------------------------------------------------"
echo "⏱ Total Training Time: ${H}h ${M}m ${S}s"
echo "--------------------------------------------------------"

echo "=== [JOB FINISHED] $(date) ==="
