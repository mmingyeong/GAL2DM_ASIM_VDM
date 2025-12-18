#!/bin/bash
#SBATCH -J vdm_predict
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm/%x.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# -------- Environment --------
module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_MODULE_LOADING=LAZY
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 65535

# -------- Paths (single source of truth) --------
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_VDM"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"
SLURM_DIR="${PROJECT_ROOT}/logs/slurm"

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
OUT_DIR_BASE="${PROJECT_ROOT}/results/vdm_predictions/${RUN_ID}"

mkdir -p "${SLURM_DIR}" "${LOG_DIR}" "${OUT_DIR_BASE}"

# Logger will only write under PROJECT_ROOT/logs/RUN_ID
export GAL2DM_LOGDIR="${LOG_DIR}"

# Always run from project root
cd "${PROJECT_ROOT}"

# -------- System info --------
echo "=== [PREDICT START] $(date) on $(hostname) ==="
which python
python - <<'PY'
import torch, os, platform
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA (Torch):", getattr(torch.version, "cuda", None))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name[0]:", torch.cuda.get_device_name(0))
print("GAL2DM_LOGDIR:", os.environ.get("GAL2DM_LOGDIR"))
PY
nvidia-smi || echo "nvidia-smi not available"

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.stderr.write("[FATAL] CUDA not available.\n")
    sys.exit(2)
PY

# -------- Checkpoint 선택 로직 --------
# 1) MODEL_PATH가 주어지면 그대로 사용
# 2) 아니면 CKPT_DIR (run 디렉토리)를 쓰고,
# 3) 그것도 없으면 results/vdm/* 중 가장 최신 디렉토리 사용

if [ -n "${MODEL_PATH:-}" ]; then
  # MODEL_PATH가 직접 지정된 경우
  if [ ! -f "${MODEL_PATH}" ]; then
    echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
    exit 1
  fi
  CKPT_DIR="$(dirname "${MODEL_PATH}")"
else
  # MODEL_PATH가 없는 경우: CKPT_DIR 기준으로 선택
  if [ -z "${CKPT_DIR:-}" ]; then
    # results/vdm/* 중 가장 최신 디렉토리 선택 (디렉토리만)
    CKPT_DIR="$(ls -dt ${PROJECT_ROOT}/results/vdm/*/ 2>/dev/null | head -n 1 || true)"
    if [ -z "${CKPT_DIR}" ]; then
      echo "[ERROR] No checkpoint directory found under ${PROJECT_ROOT}/results/vdm/"
      exit 1
    fi
  fi

  if [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR not found or not a directory: ${CKPT_DIR}"
    exit 1
  fi

  # 해당 run 디렉토리 안에서 *best*.pt 우선, 없으면 최신 .pt
  BEST_CKPT="$(ls -t ${CKPT_DIR}/*best*.pt 2>/dev/null | head -n 1 || true)"
  if [ -n "${BEST_CKPT}" ]; then
    MODEL_PATH="${BEST_CKPT}"
  else
    MODEL_PATH="$(ls -t ${CKPT_DIR}/*.pt 2>/dev/null | head -n 1 || true)"
  fi

  if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] No .pt file found in ${CKPT_DIR}"
    exit 1
  fi
fi

echo "[INFO] Using checkpoint: ${MODEL_PATH}"
echo "[INFO] CKPT_DIR        : ${CKPT_DIR}"

# Output subdir name mirrors checkpoint run stem (ex: 71897)
RUN_STEM="$(basename "${CKPT_DIR}")"
PRED_OUT_DIR="${OUT_DIR_BASE}/${RUN_STEM}"
mkdir -p "${PRED_OUT_DIR}"

# -------- Inference configuration --------
BATCH_SIZE="${BATCH_SIZE:-1}"
AMP_FLAG="--amp"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-1.0}"

# VDM + CUNet3D: 입력은 (ngal, vpec) 2ch, train과 동일
EXTRA_INPUT_FLAGS="--input_case both --keep_two_channels"

# Single console log under PROJECT_ROOT/logs/RUN_ID
MODEL_NAME="vdm_cunet3d"
CONSOLE_LOG="${LOG_DIR}/${MODEL_NAME}_predict.log"
touch "${CONSOLE_LOG}"

# -------- Run prediction (VDM version) --------
# predict 코드가 src/predict.py라면:
srun --ntasks=1 python -u -m src.predict \
  --yaml_path "${YAML_PATH}" \
  --output_dir "${PRED_OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device "cuda" \
  --batch_size "${BATCH_SIZE}" \
  --sample_fraction "${SAMPLE_FRACTION}" \
  ${AMP_FLAG} \
  ${EXTRA_INPUT_FLAGS} \
  2>&1 | tee -a "${CONSOLE_LOG}"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== [PREDICT END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}
