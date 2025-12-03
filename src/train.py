"""
train_vdm.py (VDM: Variational Diffusion Model for 3D Voxel Fields, A-SIM 128^3)
Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-07-30 | Last-Modified: 2025-11-26
"""

from __future__ import annotations
import sys
import os
import argparse
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.data_loader import get_dataloader
from src.logger import get_logger
from src.model import VDM, CUNet   # ✅ VDM + CUNet backbone


# ----------------------------
# Utilities
# ----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_clr_scheduler(optimizer, min_lr: float, max_lr: float, cycle_length: int = 8):
    """Epoch-wise triangular cyclical LR"""
    assert max_lr >= min_lr > 0
    assert cycle_length >= 2

    def triangular_clr(epoch: int):
        mid = cycle_length // 2
        ep = epoch % cycle_length
        scale = ep / max(1, mid) if ep <= mid else (cycle_length - ep) / max(1, mid)
        return (min_lr / max_lr) + (1.0 - (min_lr / max_lr)) * scale

    for pg in optimizer.param_groups:
        pg["lr"] = max_lr
    return LambdaLR(optimizer, lr_lambda=triangular_clr)


def str2bool(v):
    """Utility to parse boolean CLI args (e.g. --validate_keys False)."""
    return str(v).lower() in ("1", "true", "t", "yes", "y")


# ----------------------------
# Input selection helper
# ----------------------------
def select_inputs(x: torch.Tensor, case: str, keep_two: bool) -> torch.Tensor:
    """
    x: [B,2,D,H,W], channels=[ngal, vpec]
    case: "both" | "ch1" | "ch2"
    keep_two=True  -> 항상 2채널 반환(결측 채널은 0으로 패딩)
    keep_two=False -> 단일 채널 반환
    """
    assert x.ndim == 5 and x.size(1) == 2, f"Expected [B,2,D,H,W], got {tuple(x.shape)}"
    if case == "both":
        return x
    if case == "ch1":
        if keep_two:
            ch1 = x[:, 0:1]
            z = torch.zeros_like(ch1)
            return torch.cat([ch1, z], dim=1)
        else:
            return x[:, 0:1]
    if case == "ch2":
        if keep_two:
            ch2 = x[:, 1:2]
            z = torch.zeros_like(ch2)
            return torch.cat([z, ch2], dim=1)
        else:
            return x[:, 1:2]
    raise ValueError(f"Unknown input case: {case}")


# ----------------------------
# Train
# ----------------------------
def train(args):
    logger = get_logger("train_vdm")
    set_seed(args.seed, deterministic=args.deterministic)
    logger.info("🚀 Starting VDM training (Variational Diffusion Model) for 3D voxel fields")
    logger.info(f"Args: {vars(args)}")

    # ---- Data ----
    train_loader = get_dataloader(
        yaml_path=args.yaml_path,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        target_field=args.target_field,
        train_val_split=args.train_val_split,
        sample_fraction=args.sample_fraction,
        dtype=torch.float32,
        seed=args.seed,
        validate_keys=args.validate_keys,
        strict=False,
        exclude_list_path=args.exclude_list,
        include_list_path=args.include_list,
    )
    val_loader = get_dataloader(
        yaml_path=args.yaml_path,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        target_field=args.target_field,
        train_val_split=args.train_val_split,
        sample_fraction=1.0,
        dtype=torch.float32,
        seed=args.seed,
        validate_keys=args.validate_keys,
        strict=False,
        exclude_list_path=args.exclude_list,
        include_list_path=args.include_list,
    )

    logger.info(f"📊 Train samples (files): {len(train_loader.dataset)}")
    logger.info(f"📊 Validation samples (files): {len(val_loader.dataset)}")

    # ---- Infer field shape from one batch ----
    # y: [B, 1, D, H, W]
    sample_batch = next(iter(train_loader))
    _, y_sample = sample_batch
    assert y_sample.ndim == 5, f"Expected y shape [B,1,D,H,W], got {tuple(y_sample.shape)}"
    _, c_y, *spatial_shape = y_sample.shape
    assert c_y == 1, f"Target field should have 1 channel, got {c_y}"
    score_in_ch = 1
    score_out_ch = 1
    score_shape = (score_in_ch, *spatial_shape)  # e.g. (1, 128, 128, 128)

    # ---- Decide how many conditioning channels to expect ----
    if args.input_case == "both" or args.keep_two_channels:
        s_cond_ch = 2   # [ngal, vpec] or padded to 2
    else:
        s_cond_ch = 1   # single-channel conditioning

    # ---- Score model: CUNet backbone ----
    score_model = CUNet(
        shape=score_shape,
        out_channels=score_out_ch,
        s_conditioning_channels=s_cond_ch,
        v_conditioning_dims=[],       # 현재는 vector conditioning 사용 안 함
        v_conditioning_type="common_zerolinear",
        v_embedding_dim=64,
        v_augment=False,
        v_embed_no_s_gelu=False,
        t_conditioning=True,          # 🔑 VDM에서 항상 t를 넘기므로 True 필수
        t_embedding_dim=64,
        init_scale=0.02,
        num_res_blocks=1,
        norm_groups=8,
        mid_attn=False,               # 3D에서는 attention 비활성화 권장
        n_attention_heads=4,
        dropout_prob=0.1,
        conv_padding_mode="zeros",
        verbose=0,
    ).to(args.device)

    # ---- VDM wrapper ----
    model = VDM(
        score_model=score_model,
        noise_schedule=args.noise_schedule,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        antithetic_time_sampling=args.antithetic_time_sampling,
        data_noise=args.data_noise,
        p_cfg=args.p_cfg,
        w_cfg=args.w_cfg,
    ).to(args.device)

    logger.info(
        "🧱 Model created: VDM(score_model=CUNet(shape=%s, out_channels=%d, s_cond_ch=%d), "
        "noise_schedule=%s, gamma=[%.2f, %.2f]) | input_case=%s, keep_two=%s",
        score_shape,
        score_out_ch,
        s_cond_ch,
        args.noise_schedule,
        args.gamma_min,
        args.gamma_max,
        args.input_case,
        args.keep_two_channels,
    )

    # ---- Optimizer / Scheduler / AMP ----
    use_amp = args.amp and str(args.device).startswith("cuda")
    optimizer = Adam(model.parameters(), lr=args.max_lr)
    scheduler = get_clr_scheduler(optimizer, args.min_lr, args.max_lr, args.cycle_length)
    early_stopper = EarlyStopping(patience=args.patience, delta=args.es_delta)

    # ✅ AMP Compatibility Wrapper
    try:
        import torch.amp as amp

        scaler = amp.GradScaler("cuda") if use_amp else amp.GradScaler(enabled=False)

        def amp_autocast():
            if not use_amp:
                from contextlib import nullcontext
                return nullcontext()
            return amp.autocast("cuda", dtype=torch.float16)

    except Exception:
        from torch.cuda.amp import GradScaler as OldScaler
        from torch.cuda.amp import autocast as old_autocast

        scaler = OldScaler(enabled=use_amp)

        def amp_autocast():
            return old_autocast(enabled=use_amp)

    # ---- Paths ----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    sample_percent = int(args.sample_fraction * 100)
    case_tag = f"icase-{args.input_case}{'-keep2' if args.keep_two_channels else ''}"
    model_prefix = (
        f"{case_tag}_vdm_tgt-{args.target_field}_"
        f"bs{args.batch_size}_clr[{args.min_lr:.0e}-{args.max_lr:.0e}]_"
        f"s{args.seed}_smp{sample_percent}"
    )
    best_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_best.pt")
    final_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_final.pt")
    log_path = os.path.join(args.ckpt_dir, f"{model_prefix}_log.csv")

    # ----------------------------
    # VDM loss helper
    # ----------------------------
    def compute_vdm_loss(model, x_cond, y, args):
        """
        x_cond: 입력 (ngal, vpec) 또는 그 subset.
                -> score model의 s_conditioning으로 항상 전달 (조건부 VDM).
        y     : target field (rho or tscphi) = 데이터 x_0
        """
        loss, metrics = model.get_loss(
            x=y,
            reduction="mean",
            s_conditioning=x_cond,  # 🔑 항상 conditioning으로 사용
        )
        return loss, metrics

    # ---- Loop ----
    log_records = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        logger.info(f"🔁 Epoch {epoch + 1}/{args.epochs} started.")
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]")

        for step, (x, y) in enumerate(loop):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            # 입력 채널 선택 (ngal / vpec / both)
            # → 선택된 버전이 그대로 conditioning으로 들어감.
            x_cond = select_inputs(x, args.input_case, args.keep_two_channels)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                loss, metrics = compute_vdm_loss(model, x_cond, y, args)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * x.size(0)

            if step % max(1, args.log_interval) == 0:
                postfix = {"loss": f"{loss.item():.5f}"}
                if "elbo" in metrics:
                    postfix["elbo"] = f"{metrics['elbo'].item():.5f}"
                if "diffusion_loss" in metrics:
                    postfix["diff"] = f"{metrics['diffusion_loss'].item():.5f}"
                loop.set_postfix(**postfix)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        logger.info(f"📊 Avg Train Loss (VDM total loss): {avg_train_loss:.6f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(args.device, non_blocking=True)
                y_val = y_val.to(args.device, non_blocking=True)
                x_val_cond = select_inputs(x_val, args.input_case, args.keep_two_channels)
                with amp_autocast():
                    loss_val, metrics_val = compute_vdm_loss(model, x_val_cond, y_val, args)
                val_loss += loss_val.item() * x_val.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"📉 Epoch {epoch + 1:03d} | Val Loss (VDM total loss): {avg_val_loss:.6f} | LR: {current_lr:.2e}"
        )

        log_records.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✅ New best VDM model saved (epoch {epoch + 1})")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            logger.warning(f"🛑 Early stopping at epoch {epoch + 1}")
            break

    # ---- Save ----
    torch.save(model.state_dict(), final_model_path)
    pd.DataFrame(log_records).to_csv(log_path, index=False)
    logger.info(f"📦 Final VDM model saved: {final_model_path}")
    logger.info(f"📝 Training log saved: {log_path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VDM (Variational Diffusion Model) on A-SIM 3D fields."
    )
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument(
        "--target_field", type=str, choices=["rho", "tscphi"], default="rho"
    )
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--cycle_length", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--es_delta", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--ckpt_dir", type=str, default="results/vdm/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # 입력 채널 실험 옵션: 어느 채널을 conditioning으로 쓸지 선택
    parser.add_argument(
        "--input_case",
        type=str,
        choices=["both", "ch1", "ch2"],
        default="both",
        help="Select which input channels are provided as conditioning (ngal/vpec/both).",
    )
    parser.add_argument(
        "--keep_two_channels",
        action="store_true",
        help="If set, keep in_channels=2 and zero-pad the missing channel for single-channel cases.",
    )

    # Validation & file filtering
    parser.add_argument(
        "--validate_keys",
        type=str2bool,
        default=True,
        help="Pre-scan HDF5 to check required keys (input/output_*). Set False to skip (faster).",
    )
    parser.add_argument(
        "--exclude_list",
        type=str,
        default=None,
        help="Path to text file containing bad HDF5 file paths to exclude.",
    )
    parser.add_argument(
        "--include_list",
        type=str,
        default=None,
        help="Path to text file containing good HDF5 file paths to include only.",
    )

    # VDM-specific options
    parser.add_argument(
        "--noise_schedule",
        type=str,
        choices=["fixed_linear", "learned_linear", "learned_nn", "sigmoid"],
        default="fixed_linear",
        help="Noise schedule type used in VDM.",
    )
    parser.add_argument("--gamma_min", type=float, default=-13.3)
    parser.add_argument("--gamma_max", type=float, default=5.0)
    parser.add_argument(
        "--antithetic_time_sampling",
        type=str2bool,
        default=True,
        help="Use antithetic time sampling in VDM training.",
    )
    parser.add_argument(
        "--data_noise",
        type=float,
        default=1.0e-3,
        help="Observation noise std used in reconstruction loss term.",
    )
    parser.add_argument(
        "--p_cfg",
        type=float,
        default=None,
        help="Classifier-free guidance drop prob during training (optional; requires v_conditionings).",
    )
    parser.add_argument(
        "--w_cfg",
        type=float,
        default=None,
        help="Classifier-free guidance strength during sampling (optional; requires v_conditionings).",
    )

    args = parser.parse_args()

    try:
        train(args)
    except Exception:
        import traceback

        print("🔥 VDM training failed due to exception:")
        traceback.print_exc()
        sys.exit(1)
