#!/bin/bash
#SBATCH -J data_check_hist
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm/%x.%j.err

#SBATCH -p a40
#SBATCH --gres=gpu:A40:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-12:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# --- IMPORTANT: Slurm stdout/err 디렉토리는 잡 시작 전에 있어야 함 ---
mkdir -p /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm

# (옵션) 작업 폴더에도 백업 로그 남기고 싶으면 여기 사용
WORKDIR="/home/mingyeong/GAL2DM_ASIM_VDM/debug"
mkdir -p "${WORKDIR}/logs"

# 에러 시 어느 줄에서 죽었는지 남김
trap 'echo "[FATAL] Error on line $LINENO"; exit 1' ERR

echo "========== SLURM JOB INFO =========="
echo "Host              : $(hostname)"
echo "Date              : $(date)"
echo "JobID             : ${SLURM_JOB_ID}"
echo "JobName           : ${SLURM_JOB_NAME}"
echo "Partition         : ${SLURM_JOB_PARTITION}"
echo "SubmitDir         : ${SLURM_SUBMIT_DIR:-unknown}"
echo "CpusPerTask       : ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "===================================="

module purge
module load cuda/12.1.1

source ~/.bashrc
conda activate torch

# (권장) OpenMP/MKL thread 제한
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# (선택) HDF5 file locking 이슈 회피
export HDF5_USE_FILE_LOCKING=FALSE

# 파이썬 출력 버퍼링 방지 (print/tqdm 로그가 바로바로 찍히게)
export PYTHONUNBUFFERED=1

# 실행 위치 고정 (data_check.py가 있는 곳)
cd "${WORKDIR}"
echo "Now in WORKDIR     : $(pwd)"
echo "Python              : $(which python)"
python -c "import sys; print('Python version:', sys.version)"

# --- 실행: Slurm 로그 외에 WORKDIR/logs에도 백업 로그 남기기(추천) ---
# tee를 쓰면 Slurm .out/.err에도 가고, 아래 파일에도 추가로 저장됩니다.
# (tqdm 진행바가 stderr로 나와도 err 로그에 남습니다)
python -u 01_data_check.py 2>&1 | tee -a "${WORKDIR}/logs/data_check_${SLURM_JOB_ID}.log"

echo "Job finished on: $(date)"
