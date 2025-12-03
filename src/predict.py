# models/vdm/predict_vdm.py (or replace existing predict.py)
# -*- coding: utf-8 -*-
"""
Predict with Variational Diffusion Model (VDM + CUNet) and save to HDF5.

- Uses CUNet as score network inside VDM.
- Conditioning is given by (ngal, vpec) or its channel subsets.
- Supports channel-ablation inference:
    --input_case {both,ch1,ch2}
    --keep_two_channels  (keep s_conditioning_channels=2 and zero-pad missing channel)

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-07-30
Last-Modified: 2025-11-26
"""

from __future__ import annotations
import sys
import os
import argparse
import yaml
import torch
import h5py
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from tqdm import tqdm
from glob import glob
from contextlib import nullcontext

from src.model import VDM, CUNet
from src.logger import get_logger

logger = get_logger("predict_vdm")


# ----------------------------
# Helpers
# ----------------------------
def _natkey(path: str):
    import re
    tokens = re.split(r"(\d+)", os.path.basename(path))
    return tuple(int(t) if t.isdigit() else t for t in tokens)


def _load_yaml(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_test_files(yaml_cfg: dict) -> list[str]:
    base = yaml_cfg["asim_datasets_hdf5"]["base_path"]
    # 여기서는 validation 세트를 test 용으로 사용 (기존 UNet 스크립트와 동일한 설정)
    test_rel = yaml_cfg["asim_datasets_hdf5"]["validation_set"]["path"]
    pattern = os.path.join(base, test_rel)
    files = sorted(glob(pattern), key=_natkey)
    if not files:
        raise FileNotFoundError(f"No test HDF5 files matched: {pattern}")
    return files


def _ensure_input_shape(x: np.ndarray) -> np.ndarray:
    """
    Normalize to (N,C,D,H,W) with C in {1,2}.
    Accepts: (C,D,H,W), (N,C,D,H,W), etc.
    """
    a = np.asarray(x)
    # squeeze leading singleton dims but keep spatial dims
    while a.ndim > 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 4 and a.shape[0] in (1, 2):      # (C,D,H,W)
        a = a[None, ...]                           # (1,C,D,H,W)
    elif a.ndim == 5 and a.shape[1] in (1, 2):     # (N,C,D,H,W)
        pass
    else:
        raise ValueError(f"Unsupported 'input' shape: {a.shape}")
    return a


def _load_checkpoint(model_path: str, device: torch.device):
    """Safe checkpoint load (supports plain state_dict or wrapped)."""
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)  # type: ignore
    except TypeError:
        state = torch.load(model_path, map_location=device)
    except Exception as e:
        logger.warning(f"weights_only load failed with {e}; falling back to standard torch.load")
        state = torch.load(model_path, map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    return state


def _find_input_dataset(h5: h5py.File) -> str | None:
    """Return the dataset key to use for inputs, or None if not found."""
    candidate_keys = [
        "input", "inputs", "X",
        "data/input", "dataset/input", "features/input",
    ]
    for k in candidate_keys:
        if k in h5:
            return k
    for k in candidate_keys:
        if "/" in k:
            grp, dset = k.split("/", 1)
            if grp in h5 and dset in h5[grp]:
                return k
    lowered = {kk.lower(): kk for kk in h5.keys()}
    for name in ("input", "inputs", "x"):
        if name in lowered:
            return lowered[name]
    return None


def select_inputs(x: torch.Tensor, case: str, keep_two: bool) -> torch.Tensor:
    """
    x: [N,C,D,H,W] with C in {1,2}
    case: "both" | "ch1" | "ch2"
    keep_two=True  -> always return 2-ch with a zero-padded missing channel
    keep_two=False -> return a single channel tensor for ch1/ch2
    """
    assert x.ndim == 5 and x.size(1) in (1, 2), f"Expected [N,1or2,D,H,W], got {tuple(x.shape)}"

    if case == "both":
        if x.size(1) != 2:
            raise ValueError(f"--input_case both requires 2 channels, but got {x.size(1)}")
        return x

    if case == "ch1":
        if x.size(1) == 2:
            if keep_two:
                ch1 = x[:, 0:1]
                z   = torch.zeros_like(ch1)
                return torch.cat([ch1, z], dim=1)
            else:
                return x[:, 0:1]
        else:  # C==1 in file
            return x if not keep_two else torch.cat([x, torch.zeros_like(x)], dim=1)

    if case == "ch2":
        if x.size(1) == 2:
            if keep_two:
                ch2 = x[:, 1:2]
                z   = torch.zeros_like(ch2)
                return torch.cat([z, ch2], dim=1)
            else:
                return x[:, 1:2]
        else:  # C==1 in file
            return torch.cat([torch.zeros_like(x), x], dim=1) if keep_two else x

    raise ValueError(case)


# ----------------------------
# Inference
# ----------------------------
def run_prediction(
    yaml_path: str,
    output_dir: str,
    model_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    amp: bool = False,
    sample_fraction: float = 1.0,
    sample_seed: int = 42,
    input_case: str = "both",
    keep_two_channels: bool = False,
    on_missing_input: str = "skip",  # "skip" | "stop"
    # VDM-specific sampling options
    noise_schedule: str = "fixed_linear",
    gamma_min: float = -13.3,
    gamma_max: float = 5.0,
    antithetic_time_sampling: bool = True,
    data_noise: float = 1.0e-3,
    w_cfg: float | None = None,
    n_sampling_steps: int = 100,
):
    """
    Run inference on HDF5 test files with channel-ablation support using VDM + CUNet.

    - CUNet is used as score model.
    - VDM generates samples conditioned on (ngal, vpec) via s_conditioning.
    """
    if not (0 < sample_fraction <= 1.0):
        raise ValueError(f"--sample_fraction must be in (0,1], got {sample_fraction}")

    # case-specific subdir to avoid mixing outputs
    case_suffix = f"icase-{input_case}{'-keep2' if keep_two_channels else ''}"
    output_dir = os.path.join(output_dir, case_suffix)
    os.makedirs(output_dir, exist_ok=True)

    # RNG for reproducible file subsampling
    rng = np.random.default_rng(sample_seed)

    # 1) Resolve test set
    cfg = _load_yaml(yaml_path)
    test_files = _resolve_test_files(cfg)

    if sample_fraction < 1.0:
        n_total = len(test_files)
        n_keep = max(1, int(np.ceil(sample_fraction * n_total)))
        keep_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
        test_files = [test_files[i] for i in keep_idx]
        logger.info(f"🧪 Test files subsampled: {n_keep}/{n_total} (fraction={sample_fraction:.3f})")
    else:
        logger.info(f"🧪 Test files: {len(test_files)} found from YAML (no subsampling).")

    if len(test_files) == 0:
        logger.warning("No test files resolved; nothing to do.")
        return

    dev = torch.device(device)

    # 2) Infer spatial shape and conditioning channels from the FIRST file
    with h5py.File(test_files[0], "r") as f0:
        key0 = _find_input_dataset(f0)
        if key0 is None:
            raise KeyError(f"No input-like dataset found in first test file: {test_files[0]}")
        x0_np = f0[key0][:]
    x0_np = _ensure_input_shape(x0_np)  # (N,C,D,H,W)
    _, c0, D, H, W = x0_np.shape
    if c0 not in (1, 2):
        raise ValueError(f"First file input channels must be 1 or 2, got {c0}")

    # Decide how many conditioning channels will actually be used
    if input_case == "both" or keep_two_channels:
        s_cond_ch = 2
    else:
        s_cond_ch = 1

    # VDM latent/data space: 1-channel field with same spatial dims
    score_in_ch = 1
    score_out_ch = 1
    score_shape = (score_in_ch, D, H, W)

    # 3) Build VDM + CUNet and load checkpoint
    score_model = CUNet(
        shape=score_shape,
        out_channels=score_out_ch,
        chs=[48, 96, 192, 384],
        s_conditioning_channels=s_cond_ch,
        v_conditioning_dims=[],            # no vector conditioning in this setup
        v_conditioning_type="common_zerolinear",
        v_embedding_dim=64,
        v_augment=False,
        v_embed_no_s_gelu=False,
        t_conditioning=True,               # VDM uses time-conditioning
        t_embedding_dim=64,
        init_scale=0.02,
        num_res_blocks=1,
        norm_groups=8,
        mid_attn=False,                    # 3D에서는 attention 비활성화 권장
        n_attention_heads=4,
        dropout_prob=0.1,
        conv_padding_mode="zeros",
        verbose=0,
    ).to(dev)

    model = VDM(
        score_model=score_model,
        noise_schedule=noise_schedule,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        antithetic_time_sampling=antithetic_time_sampling,
        data_noise=data_noise,
        p_cfg=None,        # 샘플링 단계에서는 p_cfg 사용 안 함
        w_cfg=w_cfg,       # classifier-free guidance strength (if used)
    ).to(dev)

    logger.info(
        "🧱 Model: VDM(CUNet(shape=%s, out_channels=%d, s_cond_ch=%d), "
        "noise_schedule=%s, gamma=[%.2f, %.2f]) | input_case=%s, keep_two=%s",
        score_shape,
        score_out_ch,
        s_cond_ch,
        noise_schedule,
        gamma_min,
        gamma_max,
        input_case,
        keep_two_channels,
    )

    logger.info(f"📥 Loading checkpoint: {model_path}")
    state = _load_checkpoint(model_path, dev)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys while loading: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading: {unexpected}")
    model.eval()

    # AMP context
    try:
        _ = torch.amp
        if amp and dev.type == "cuda":
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        elif amp and dev.type == "cpu":
            autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()
    except Exception:
        from torch.cuda.amp import autocast as legacy_autocast
        autocast_ctx = legacy_autocast(enabled=amp)

    # 4) Predict per file (conditional sampling with VDM)
    saved_files: list[str] = []
    skipped_files: list[str] = []

    torch.set_grad_enabled(False)
    with torch.no_grad():
        for input_path in tqdm(test_files, desc="🚀 Running VDM test predictions"):
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename)

            # Load inputs (robust to missing/variant keys)
            with h5py.File(input_path, "r") as f:
                key = _find_input_dataset(f)
                if key is None:
                    msg = f"No input-like dataset found in {input_path}"
                    if on_missing_input == "stop":
                        raise KeyError(msg)
                    logger.warning(f"⚠️ {msg} — SKIP")
                    skipped_files.append(input_path)
                    continue

                try:
                    x_np = f[key][:]
                except Exception as e:
                    msg = f"Failed to read dataset '{key}' in {input_path}: {e}"
                    if on_missing_input == "stop":
                        raise RuntimeError(msg)
                    logger.warning(f"⚠️ {msg} — SKIP")
                    skipped_files.append(input_path)
                    continue

            # Normalize to (N,C,D,H,W)
            try:
                x_np = _ensure_input_shape(x_np)
            except Exception as e:
                msg = f"Invalid input shape in {input_path}: {e}"
                if on_missing_input == "stop":
                    raise
                logger.warning(f"⚠️ {msg} — SKIP")
                skipped_files.append(input_path)
                continue

            # To tensor & select conditioning channels
            x_tensor = torch.from_numpy(np.ascontiguousarray(x_np)).float().to(dev)
            try:
                x_cond = select_inputs(x_tensor, input_case, keep_two_channels)  # [N,s_cond_ch,D,H,W]
            except Exception as e:
                msg = f"Channel selection failed for {input_path}: {e}"
                if on_missing_input == "stop":
                    raise
                logger.warning(f"⚠️ {msg} — SKIP")
                skipped_files.append(input_path)
                continue

            if x_cond.size(1) != s_cond_ch:
                msg = (
                    f"Post-selection conditioning channels {x_cond.size(1)} "
                    f"!= expected s_conditioning_channels {s_cond_ch} in {input_path}"
                )
            # Batched conditional sampling with VDM
            preds = []
            for i in range(0, x_cond.shape[0], batch_size):
                s_batch = x_cond[i: i + batch_size]  # conditioning batch
                with autocast_ctx:
                    # VDM.sample generates [B,1,D,H,W] samples from prior -> data space
                    y_batch = model.sample(
                        batch_size=s_batch.shape[0],
                        n_sampling_steps=n_sampling_steps,
                        device=device,
                        s_conditioning=s_batch,
                        verbose=False,
                    )
                preds.append(y_batch.float().cpu().numpy())

            y_pred = np.concatenate(preds, axis=0)  # (N,1,D,H,W)
            y_pred = np.squeeze(y_pred, axis=1)     # (N,D,H,W) or (D,H,W) if N==1

            # Save prediction
            with h5py.File(output_path, "w") as f_out:
                f_out.create_dataset("prediction", data=y_pred, compression="gzip")
                # Meta-info
                f_out.attrs["source_file"] = input_path
                f_out.attrs["model_path"] = model_path
                f_out.attrs["model_class"] = model.__class__.__name__
                f_out.attrs["amp"] = bool(amp)
                f_out.attrs["input_case"] = str(input_case)
                f_out.attrs["keep_two_channels"] = bool(keep_two_channels)
                f_out.attrs["noise_schedule"] = str(noise_schedule)
                f_out.attrs["gamma_min"] = float(gamma_min)
                f_out.attrs["gamma_max"] = float(gamma_max)
                f_out.attrs["n_sampling_steps"] = int(n_sampling_steps)
                if w_cfg is not None:
                    f_out.attrs["w_cfg"] = float(w_cfg)

            logger.info(f"✅ Saved: {output_path}")
            saved_files.append(output_path)

    # Summary
    logger.info("====== VDM Inference Summary ======")
    logger.info(f"Saved files : {len(saved_files)}")
    logger.info(f"Skipped     : {len(skipped_files)}")
    if skipped_files:
        logger.info("Skipped list (first 20): " + ", ".join(os.path.basename(p) for p in skipped_files[:20]))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VDM (Variational Diffusion Model) inference on A-SIM test files."
    )

    # Data / Paths
    parser.add_argument("--yaml_path", type=str, required=True,
                        help="Path to asim_paths.yaml")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to save predictions (per-case subdir will be created)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained VDM .pt checkpoint (state_dict of VDM)")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision inference")

    # Subsampling
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction (0,1] of TEST FILES to run. Does NOT subsample within-file samples.")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed for reproducible file-level subsampling.")

    # Channel ablation flags (affect s_conditioning only)
    parser.add_argument("--input_case", type=str, choices=["both", "ch1", "ch2"], default="both",
                        help="Select which input channels are provided as conditioning (ngal/vpec/both).")
    parser.add_argument("--keep_two_channels", action="store_true",
                        help="If set, keep s_conditioning_channels=2 and zero-pad missing channel for single-channel cases.")

    # Missing-input handling policy
    parser.add_argument("--on_missing_input", type=str, choices=["skip", "stop"], default="skip",
                        help="When an HDF5 file lacks the input dataset: skip it (default) or stop with error.")

    # VDM-specific hyperparameters (must match training!)
    parser.add_argument("--noise_schedule", type=str,
                        choices=["fixed_linear", "learned_linear", "learned_nn", "sigmoid"],
                        default="fixed_linear",
                        help="Noise schedule type used in VDM (should match training).")
    parser.add_argument("--gamma_min", type=float, default=-13.3,
                        help="Minimum gamma used at training (should match training).")
    parser.add_argument("--gamma_max", type=float, default=5.0,
                        help="Maximum gamma used at training (should match training).")
    parser.add_argument("--antithetic_time_sampling", type=lambda s: str(s).lower() in ("1", "true", "t", "yes", "y"),
                        default=True,
                        help="Use antithetic time sampling (should match training).")
    parser.add_argument("--data_noise", type=float, default=1.0e-3,
                        help="Observation noise std used in reconstruction loss term (training-only param; "
                             "kept for completeness).")
    parser.add_argument("--w_cfg", type=float, default=None,
                        help="Classifier-free guidance strength during sampling (if trained with p_cfg).")

    # Sampling steps
    parser.add_argument("--n_sampling_steps", type=int, default=100,
                        help="Number of diffusion sampling steps (discretization of [1,0]).")

    args = parser.parse_args()

    run_prediction(
        yaml_path=args.yaml_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        amp=args.amp,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
        input_case=args.input_case,
        keep_two_channels=args.keep_two_channels,
        on_missing_input=args.on_missing_input,
        noise_schedule=args.noise_schedule,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        antithetic_time_sampling=args.antithetic_time_sampling,
        data_noise=args.data_noise,
        w_cfg=args.w_cfg,
        n_sampling_steps=args.n_sampling_steps,
    )
