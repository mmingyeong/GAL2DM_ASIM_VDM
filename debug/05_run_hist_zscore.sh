#!/bin/bash
#SBATCH -J hist_minmax01_vr
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm/%x.%j.err
#SBATCH -p a40
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

mkdir -p /home/mingyeong/GAL2DM_ASIM_VDM/logs/slurm

module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

cd /home/mingyeong/GAL2DM_ASIM_VDM/debug
python -u 03_hist_zscore_signedlog_vpec_rho.py