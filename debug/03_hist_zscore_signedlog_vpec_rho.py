import os, glob, time, socket, logging
from logging.handlers import RotatingFileHandler
import numpy as np
import h5py
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# -----------------------
# Settings
# -----------------------
PATH_GLOB = "/gpfs/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/*.hdf5"
OUT_DIR = "./hdf5_hist_zscore_signedlog_vpec_rho"
os.makedirs(OUT_DIR, exist_ok=True)

BINS = 240
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))
TARGET_DTYPE = None  # 필요시 np.float32

# zscore 후 signed-log1p 값 range (필요하면 넓히세요)
HIST_RANGE = (-6.0, 6.0)

LOG_FILE = os.path.join(OUT_DIR, "run.log")
FAILED_FILE = os.path.join(OUT_DIR, "failed_files.txt")

# -----------------------
# Logging
# -----------------------
def setup_logger(path):
    logger = logging.getLogger("zscore_signedlog_vpec_rho")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(path, maxBytes=50*1024*1024, backupCount=3)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger(LOG_FILE)

# -----------------------
# Helpers
# -----------------------
def read_two(f):
    if "input" not in f or "output_rho" not in f:
        raise KeyError("missing 'input' or 'output_rho'")
    vpec = f["input"][0][1]
    rho  = f["output_rho"][0, 0]
    return vpec, rho

def finite_flat(x, target_dtype=None):
    a = np.asarray(x)
    if target_dtype is not None:
        a = a.astype(target_dtype, copy=False)
    flat = a.reshape(-1)
    if not np.isfinite(flat).all():
        flat = flat[np.isfinite(flat)]
    return flat

def signed_log1p(x):
    x = x.astype(np.float64, copy=False)
    return np.sign(x) * np.log1p(np.abs(x))

# -----------------------
# Pass 1: global mean/std (Welford merge)
# -----------------------
def file_stats(fp, target_dtype):
    with h5py.File(fp, "r") as f:
        vpec, rho = read_two(f)

        def stats(arr):
            arr = finite_flat(arr, target_dtype)
            n = arr.size
            if n == 0:
                return (0, 0.0, 0.0)  # n, mean, M2
            arr = arr.astype(np.float64, copy=False)
            mean = float(arr.mean())
            M2 = float(((arr - mean) ** 2).sum())
            return (n, mean, M2)

        return stats(vpec), stats(rho)

def merge_welford(a, b):
    n1, m1, M21 = a
    n2, m2, M22 = b
    if n1 == 0: return (n2, m2, M22)
    if n2 == 0: return (n1, m1, M21)
    n = n1 + n2
    delta = m2 - m1
    mean = m1 + delta * (n2 / n)
    M2 = M21 + M22 + delta*delta * (n1*n2 / n)
    return (n, mean, M2)

def finalize(acc):
    n, mean, M2 = acc
    if n <= 1:
        return (mean, 0.0)
    var = M2 / (n - 1)
    return (mean, float(np.sqrt(var)))

# -----------------------
# Pass 2: histogram accumulation (zscore + signed-log)
# -----------------------
def file_hist_zlog(fp, bins, hist_range, v_mu, v_sd, r_mu, r_sd, target_dtype):
    with h5py.File(fp, "r") as f:
        vpec, rho = read_two(f)
        v = finite_flat(vpec, target_dtype)
        r = finite_flat(rho,  target_dtype)

        def zlog(arr, mu, sd):
            if arr.size == 0:
                return arr
            arr = arr.astype(np.float64, copy=False)
            if not np.isfinite(sd) or sd == 0.0:
                zz = np.zeros_like(arr)
            else:
                zz = (arr - mu) / sd
            return signed_log1p(zz)

        vz = zlog(v, v_mu, v_sd)
        rz = zlog(r, r_mu, r_sd)

        hv, _ = np.histogram(vz, bins=bins, range=hist_range)
        hr, _ = np.histogram(rz, bins=bins, range=hist_range)

    return hv.astype(np.int64), hr.astype(np.int64)

def plot_hist(counts, edges, title, out_path, log_y=True):
    centers = 0.5*(edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    plt.figure(figsize=(8,5))
    plt.bar(centers, counts, width=widths, align="center")
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    if log_y:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -----------------------
# Main
# -----------------------
t0 = time.time()
logger.info("===== Case2 Z-score + signed-log (vpec, rho) started =====")
logger.info(f"Host={socket.gethostname()} PID={os.getpid()}")
logger.info(f"PATH_GLOB={PATH_GLOB}")
logger.info(f"OUT_DIR={os.path.abspath(OUT_DIR)}")
logger.info(f"BINS={BINS} WORKERS={WORKERS} TARGET_DTYPE={TARGET_DTYPE}")
logger.info(f"HIST_RANGE={HIST_RANGE}")

files = sorted(glob.glob(PATH_GLOB))
if not files:
    raise FileNotFoundError(f"No files matched: {PATH_GLOB}")
logger.info(f"Found {len(files)} files")

failed = []

# Pass1
logger.info("Pass1: computing global mean/std for vpec, rho ...")
v_acc = (0, 0.0, 0.0)
r_acc = (0, 0.0, 0.0)

pbar = tqdm(total=len(files), desc="Pass1(stats)", unit="file", dynamic_ncols=True) if tqdm else None
with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    fut_to_fp = {ex.submit(file_stats, fp, TARGET_DTYPE): fp for fp in files}
    for fut in as_completed(fut_to_fp):
        fp = fut_to_fp[fut]
        try:
            vs, rs = fut.result()
            v_acc = merge_welford(v_acc, vs)
            r_acc = merge_welford(r_acc, rs)
        except Exception as e:
            failed.append(fp)
            logger.warning(f"[PASS1-SKIP] {fp} -> {repr(e)}")
        if pbar: pbar.update(1)
if pbar: pbar.close()

v_mu, v_sd = finalize(v_acc)
r_mu, r_sd = finalize(r_acc)

logger.info(f"Global stats: vpec mean={v_mu} std={v_sd} (n={v_acc[0]})")
logger.info(f"Global stats: rho  mean={r_mu} std={r_sd} (n={r_acc[0]})")

# Pass2
logger.info("Pass2: accumulating histograms on zscore+signed-log values ...")
vpec_counts = np.zeros(BINS, dtype=np.int64)
rho_counts  = np.zeros(BINS, dtype=np.int64)

badset = set(failed)
pbar = tqdm(total=len(files) - len(badset), desc="Pass2(hist)", unit="file", dynamic_ncols=True) if tqdm else None

with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    fut_to_fp = {
        ex.submit(file_hist_zlog, fp, BINS, HIST_RANGE, v_mu, v_sd, r_mu, r_sd, TARGET_DTYPE): fp
        for fp in files if fp not in badset
    }
    for fut in as_completed(fut_to_fp):
        fp = fut_to_fp[fut]
        try:
            hv, hr = fut.result()
            vpec_counts += hv
            rho_counts  += hr
        except Exception as e:
            failed.append(fp)
            logger.warning(f"[PASS2-SKIP] {fp} -> {repr(e)}")
        if pbar: pbar.update(1)
if pbar: pbar.close()

failed = sorted(set(failed))
if failed:
    with open(FAILED_FILE, "w") as f:
        f.write("\n".join(failed) + "\n")
    logger.warning(f"Failed files: {len(failed)} -> {FAILED_FILE}")

edges = np.linspace(HIST_RANGE[0], HIST_RANGE[1], BINS+1)

logger.info("Saving plots ...")
plot_hist(vpec_counts, edges, "vpec zscore -> signedlog1p(z)", os.path.join(OUT_DIR, "hist_vpec_zlog.png"))
plot_hist(rho_counts,  edges, "rho  zscore -> signedlog1p(z)", os.path.join(OUT_DIR, "hist_rho_zlog.png"))

fig, axes = plt.subplots(1,2, figsize=(12,5))
for ax, counts, name in [(axes[0], vpec_counts, "vpec"), (axes[1], rho_counts, "rho")]:
    centers = 0.5*(edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    ax.bar(centers, counts, width=widths, align="center")
    ax.set_title(f"{name} z->signedlog1p(z)")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_all_zlog.png"), dpi=200, bbox_inches="tight")
plt.close()

logger.info(f"[OK] Saved to {os.path.abspath(OUT_DIR)}")
logger.info(f"Elapsed {(time.time()-t0)/60:.2f} min")
logger.info("===== Finished =====")
