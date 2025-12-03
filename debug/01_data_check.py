import os
import glob
import time
import socket
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import h5py
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# -----------------------
# Settings
# -----------------------
path = "/gpfs/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/*.hdf5"
out_dir = "./hdf5_hist_outputs_fast"
os.makedirs(out_dir, exist_ok=True)

bins = 200
log_y = True

# ★ 중요: 샘플링 없이 1-pass 하려면 범위를 고정해야 함 (데이터에 맞게 조정)
RANGE_NGAL = (0.0, 2.0)       # 예시
RANGE_VPEC = (-5000.0, 5000.0) # 예시
RANGE_RHO  = (0.0, 800.0)     # 예시

# 병렬 워커 수
workers = 6

# histogram에는 보통 float32면 충분
target_dtype = None  # None이면 변환 안 함, 예: np.float32

# logging options
log_file = os.path.join(out_dir, "run.log")
failed_list_file = os.path.join(out_dir, "failed_files.txt")
log_level = logging.INFO

# -----------------------
# Logging setup
# -----------------------
def setup_logger(log_path: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("data_check")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger(log_file, log_level)

logger.info("===== Job started =====")
logger.info(f"Host: {socket.gethostname()}")
logger.info(f"PID: {os.getpid()}")
logger.info(f"path glob: {path}")
logger.info(f"out_dir: {os.path.abspath(out_dir)}")
logger.info(f"bins={bins}, log_y={log_y}, workers={workers}, target_dtype={target_dtype}")
logger.info(f"Ranges: ngal={RANGE_NGAL}, vpec={RANGE_VPEC}, rho={RANGE_RHO}")
logger.info(f"numpy={np.__version__}, h5py={h5py.__version__}")

if tqdm is None:
    logger.warning("tqdm not installed. Progress bar will be disabled. (pip/conda install tqdm)")

# -----------------------
# stats helper
# -----------------------
def _safe_stats(x: np.ndarray):
    """Return (n, min, max, sum, sumsq) for finite values only."""
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0, np.nan, np.nan, 0.0, 0.0
    n = int(x.size)
    mn = float(x.min())
    mx = float(x.max())
    s = float(x.sum(dtype=np.float64))
    ss = float(np.dot(x.astype(np.float64, copy=False), x.astype(np.float64, copy=False)))  # sum of squares
    return n, mn, mx, s, ss

def _merge_stats(acc, part):
    """acc/part: (n, min, max, sum, sumsq)"""
    n0, mn0, mx0, s0, ss0 = acc
    n1, mn1, mx1, s1, ss1 = part

    n = n0 + n1
    mn = mn0 if (n0 > 0 and (n1 == 0 or mn0 <= mn1)) else mn1
    mx = mx0 if (n0 > 0 and (n1 == 0 or mx0 >= mx1)) else mx1

    if n0 == 0 and n1 == 0:
        mn, mx = np.nan, np.nan

    return (n, mn, mx, s0 + s1, ss0 + ss1)

def _finalize_stats(st):
    """(n, min, max, sum, sumsq) -> dict(min,max,mean,std)"""
    n, mn, mx, s, ss = st
    if n == 0:
        return {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    mean = s / n
    var = ss / n - mean * mean
    if var < 0 and var > -1e-12:  # numeric guard
        var = 0.0
    std = float(np.sqrt(var))
    return {"n": n, "min": mn, "max": mx, "mean": float(mean), "std": std}

# -----------------------
# Worker: per-file histogram + stats
# -----------------------
def file_hists(fp, bins, r_ngal, r_vpec, r_rho, target_dtype):
    with h5py.File(fp, "r") as f:
        ngal = f["input"][0][0]
        vpec = f["input"][0][1]
        rho  = f["output_rho"][0, 0]

        if target_dtype is not None:
            ngal = ngal.astype(target_dtype, copy=False)
            vpec = vpec.astype(target_dtype, copy=False)
            rho  = rho.astype(target_dtype,  copy=False)

        ngal_arr = np.asarray(ngal)
        vpec_arr = np.asarray(vpec)
        rho_arr  = np.asarray(rho)

        # flatten + finite guard
        ngal_flat = ngal_arr.reshape(-1)
        vpec_flat = vpec_arr.reshape(-1)
        rho_flat  = rho_arr.reshape(-1)

        if not np.isfinite(ngal_flat).all():
            ngal_flat = ngal_flat[np.isfinite(ngal_flat)]
        if not np.isfinite(vpec_flat).all():
            vpec_flat = vpec_flat[np.isfinite(vpec_flat)]
        if not np.isfinite(rho_flat).all():
            rho_flat = rho_flat[np.isfinite(rho_flat)]

        h_ngal, _ = np.histogram(ngal_flat, bins=bins, range=r_ngal)
        h_vpec, _ = np.histogram(vpec_flat, bins=bins, range=r_vpec)
        h_rho,  _ = np.histogram(rho_flat,  bins=bins, range=r_rho)

        # stats (min/max/mean/std를 위해 count/sum/sumsq까지)
        st_ngal = _safe_stats(ngal_flat)
        st_vpec = _safe_stats(vpec_flat)
        st_rho  = _safe_stats(rho_flat)

    return (
        h_ngal.astype(np.int64), h_vpec.astype(np.int64), h_rho.astype(np.int64),
        st_ngal, st_vpec, st_rho
    )

# -----------------------
# Main accumulate
# -----------------------
t0 = time.time()
files = sorted(glob.glob(path))
if not files:
    raise FileNotFoundError(f"No files matched: {path}")

logger.info(f"Found {len(files)} files.")

ngal_counts = np.zeros(bins, dtype=np.int64)
vpec_counts = np.zeros(bins, dtype=np.int64)
rho_counts  = np.zeros(bins, dtype=np.int64)

# overall stats accumulators: (n, min, max, sum, sumsq)
ngal_stats_acc = (0, np.nan, np.nan, 0.0, 0.0)
vpec_stats_acc = (0, np.nan, np.nan, 0.0, 0.0)
rho_stats_acc  = (0, np.nan, np.nan, 0.0, 0.0)

failed = []

pbar = tqdm(total=len(files), desc="Processing files", unit="file", dynamic_ncols=True) if tqdm is not None else None

with ProcessPoolExecutor(max_workers=workers) as ex:
    fut_to_fp = {
        ex.submit(file_hists, fp, bins, RANGE_NGAL, RANGE_VPEC, RANGE_RHO, target_dtype): fp
        for fp in files
    }

    last_log = time.time()
    done = 0

    for fut in as_completed(fut_to_fp):
        fp = fut_to_fp[fut]
        try:
            h_ngal, h_vpec, h_rho, st_ngal, st_vpec, st_rho = fut.result()
            ngal_counts += h_ngal
            vpec_counts += h_vpec
            rho_counts  += h_rho

            ngal_stats_acc = _merge_stats(ngal_stats_acc, st_ngal)
            vpec_stats_acc = _merge_stats(vpec_stats_acc, st_vpec)
            rho_stats_acc  = _merge_stats(rho_stats_acc,  st_rho)

        except Exception as e:
            failed.append(fp)
            logger.exception(f"[FAIL] {fp} -> {repr(e)}")

        done += 1
        if pbar is not None:
            pbar.update(1)

        now = time.time()
        if now - last_log > 30 or done == len(files):
            elapsed = now - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            remain = (len(files) - done) / rate if rate > 0 else float("inf")
            logger.info(f"Progress: {done}/{len(files)} | {rate:.2f} files/s | ETA ~ {remain/60:.1f} min")
            last_log = now

if pbar is not None:
    pbar.close()

# 실패 파일 리스트 저장
if failed:
    with open(failed_list_file, "w") as f:
        for fp in failed:
            f.write(fp + "\n")
    logger.warning(f"Failed files: {len(failed)} (saved to {failed_list_file})")
else:
    logger.info("No failed files.")

# edges 생성 (range 기반)
ngal_edges = np.linspace(RANGE_NGAL[0], RANGE_NGAL[1], bins + 1)
vpec_edges = np.linspace(RANGE_VPEC[0], RANGE_VPEC[1], bins + 1)
rho_edges  = np.linspace(RANGE_RHO[0],  RANGE_RHO[1],  bins + 1)

# finalize stats
ngal_stats = _finalize_stats(ngal_stats_acc)
vpec_stats = _finalize_stats(vpec_stats_acc)
rho_stats  = _finalize_stats(rho_stats_acc)

def _stats_text(d):
    return (
        f"n={d['n']}\n"
        f"min={d['min']:.6g}\n"
        f"max={d['max']:.6g}\n"
        f"mean={d['mean']:.6g}\n"
        f"std={d['std']:.6g}"
    )

def plot_hist(counts, edges, title, out_path, stats_dict, log_y=True):
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    plt.figure(figsize=(8, 5))
    plt.bar(centers, counts, width=widths, align="center")
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    if log_y:
        plt.yscale("log")

    # ---- add stats text box (top-right in axes coords)
    plt.gca().text(
        0.98, 0.98,
        _stats_text(stats_dict),
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

logger.info("Saving histograms...")

plot_hist(ngal_counts, ngal_edges, "ngal histogram (all files)", os.path.join(out_dir, "hist_ngal.png"), ngal_stats, log_y=log_y)
plot_hist(vpec_counts, vpec_edges, "vpec histogram (all files)", os.path.join(out_dir, "hist_vpec.png"), vpec_stats, log_y=log_y)
plot_hist(rho_counts,  rho_edges,  "rho histogram (all files)",  os.path.join(out_dir, "hist_rho.png"),  rho_stats,  log_y=log_y)

# combined
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, counts, edges, name, st in [
    (axes[0], ngal_counts, ngal_edges, "ngal", ngal_stats),
    (axes[1], vpec_counts, vpec_edges, "vpec", vpec_stats),
    (axes[2], rho_counts,  rho_edges,  "rho",  rho_stats),
]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    ax.bar(centers, counts, width=widths, align="center")
    ax.set_title(name)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    if log_y:
        ax.set_yscale("log")

    ax.text(
        0.98, 0.98,
        _stats_text(st),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="none"),
    )

plt.tight_layout()
all_path = os.path.join(out_dir, "hist_all.png")
plt.savefig(all_path, dpi=200, bbox_inches="tight")
plt.close()

logger.info(f"Saved plots into: {os.path.abspath(out_dir)}")
logger.info(f"Combined figure: {all_path}")

t1 = time.time()
logger.info(f"Total elapsed: {(t1 - t0)/60:.2f} min")
logger.info("===== Job finished =====")
