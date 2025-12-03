"""
model.py (VDM + CUNet backbone for 3D Voxel Regression)

Description:
    - Variational Diffusion Model (VDM) following Kingma et al. (2021)
    - CUNet backbone (from cfpark00/MLtools) adapted as a 2D/3D score network

Typical usage:
    from model import VDM, CUNet

    score_model = CUNet(
        shape=(1, 128, 128, 128),   # (C, D, H, W) for 3D
        out_channels=1,
        s_conditioning_channels=2,  # e.g. [ngal, vpec]
        v_conditioning_dims=[],     # no vector conditioning for now
        t_conditioning=True,        # VDM always passes t
        mid_attn=False,             # 3D: disable attention
    )

    vdm = VDM(score_model=score_model, noise_schedule="fixed_linear", ...)

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-11-18

Reference:
    - Kingma et al. (2021), "Variational Diffusion Models", arXiv:2107.00630
    - cfpark00/MLtools VDM source:
      https://github.com/cfpark00/MLtools/blob/main/mltools/models/vdm_model.py
    - Park, Core Francisco, et al. (2024),
      "3D Reconstruction of Dark Matter Fields with Diffusion Models:
       Towards Application to Galaxy Surveys",
      ICML 2024 AI4Science Workshop.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, autograd
from torch.distributions.normal import Normal
from torch.special import expm1
from tqdm import trange

from src.model_tools import (
    kl_std_normal,
    FixedLinearSchedule,
    SigmoidSchedule,
    LearnedLinearSchedule,
    NNSchedule,
    zero_init,
    get_conv,
    get_timestep_embedding,
    AttnBlock,
    ResNetBlock,
    ResNetDown,
    ResNetUp,
)

# =========================================================================
# Variational Diffusion Model (VDM)
# =========================================================================


class VDM(nn.Module):
    def __init__(
        self,
        score_model: nn.Module,
        noise_schedule: str = "fixed_linear",
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        antithetic_time_sampling: bool = True,
        data_noise: float = 1.0e-3,
        p_cfg=None,
        w_cfg=None,
    ):
        """Variational diffusion model, continuous-time implementation of arxiv:2107.00630.

        Args
        ----
        score_model:
            Score (denoising) network. Must accept forward(x, t=...) where
            t is a 1D tensor of shape (B,) in [0, 1].
        noise_schedule:
            One of {"fixed_linear", "learned_linear", "learned_nn", "sigmoid"}.
        gamma_min, gamma_max:
            Range of γ(t). These define alpha(t) and sigma(t).
        antithetic_time_sampling:
            Whether to use antithetic sampling in t for lower variance estimates.
        data_noise:
            Observation noise std used in the reconstruction loss term.
        p_cfg:
            Classifier-free guidance drop probability during training (optional).
        w_cfg:
            Classifier-free guidance strength during sampling (optional).
        """
        super().__init__()
        self.score_model = score_model
        self.data_noise = data_noise
        assert noise_schedule in [
            "fixed_linear",
            "learned_linear",
            "learned_nn",
            "sigmoid",
        ], f"Unknown noise schedule {noise_schedule}"
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(self.gamma_min, self.gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(self.gamma_min, self.gamma_max)
        elif noise_schedule == "learned_nn":
            self.gamma = NNSchedule(self.gamma_min, self.gamma_max)
        elif noise_schedule == "sigmoid":
            self.gamma = SigmoidSchedule(self.gamma_min, self.gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {noise_schedule}")

        self.antithetic_time_sampling = antithetic_time_sampling
        self.p_cfg = p_cfg
        self.w_cfg = w_cfg

    # ------------------------------------------------------------------
    # Forward SDE / variance-preserving map
    # ------------------------------------------------------------------
    def variance_preserving_map(
        self, x: Tensor, times: Tensor, noise: Optional[Tensor] = None
    ):
        """Add noise to data sample in a variance-preserving way (Eq. 10 in arxiv:2107.00630).

        Args
        ----
        x:
            Data sample x_0, shape (B, C, ...).
        times:
            Diffusion times in [0, 1], shape (B,).
        noise:
            Optional pre-sampled noise ε ~ N(0, I), same shape as x.

        Returns
        -------
        x_t:
            Noisy sample at time t.
        gamma_t:
            γ(t) with shape (B, 1, ..., 1) broadcast to x's spatial dims.
        """
        # --- 입력 x, times 기본 체크 (디버깅용) ---
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError("[NaN DETECTED] x has NaN/Inf in variance_preserving_map")
        if torch.isnan(times).any() or torch.isinf(times).any():
            raise RuntimeError(
                "[NaN DETECTED] times has NaN/Inf in variance_preserving_map"
            )

        # Need gradient w.r.t. times to compute gamma_grad in the loss
        with torch.enable_grad():
            times = times.view((times.shape[0],) + (1,) * (x.ndim - 1))
            gamma_t = self.gamma(times)

        # --- gamma_t 안정화 및 NaN/Inf 체크 ---
        if torch.isnan(gamma_t).any() or torch.isinf(gamma_t).any():
            raise RuntimeError(
                "[NaN DETECTED] gamma_t has NaN/Inf in variance_preserving_map"
            )

        # learned schedule일 경우를 대비해 살짝 margin 두고 clamp
        gamma_t = torch.clamp(
            gamma_t,
            min=self.gamma_min - 2.0,
            max=self.gamma_max + 2.0,
        )

        alpha = self.alpha(gamma_t)
        sigma = self.sigma(gamma_t)

        # --- alpha, sigma 체크 ---
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            raise RuntimeError(
                "[NaN DETECTED] alpha has NaN/Inf in variance_preserving_map"
            )
        if torch.isnan(sigma).any() or torch.isinf(sigma).any():
            raise RuntimeError(
                "[NaN DETECTED] sigma has NaN/Inf in variance_preserving_map"
            )

        # --- noise: None 이거나 NaN/Inf를 포함하면 새로 생성해서 덮어쓰기 ---
        if (noise is None) or torch.isnan(noise).any() or torch.isinf(noise).any():
            # 디버깅용으로 한 번 찍고 싶으면 print 추가해도 됨
            # print("[WARN] noise was None or had NaN/Inf — regenerating with randn_like")
            noise = torch.randn_like(x)

        x_t = alpha * x + noise * sigma

        if torch.isnan(x_t).any() or torch.isinf(x_t).any():
            raise RuntimeError(
                "[NaN DETECTED] x_t has NaN/Inf right after alpha*x + noise*sigma"
            )

        return x_t, gamma_t



    def sample_zt_given_zs(
        self, zs: Tensor, s: Tensor, t: Tensor, pos_mean: bool = False
    ) -> Tensor:
        """Sample z_t given z_s with the forward SDE transition."""
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        alpha_t = self.alpha(gamma_t)
        alpha_s = self.alpha(gamma_s)
        sigma_t = self.sigma(gamma_t)
        sigma_s = self.sigma(gamma_s)

        alpha_ts = alpha_t / alpha_s
        sigma_ts_sq = sigma_t**2 - (alpha_ts**2) * (sigma_s**2)

        if pos_mean:
            return alpha_ts * zs

        return alpha_ts * zs + torch.sqrt(sigma_ts_sq) * torch.randn_like(zs)

    def sample_times(self, batch_size: int, device: str) -> Tensor:
        """Sample diffusion times for a batch, used for Monte Carlo estimates.

        Returns
        -------
        times:
            Uniform samples in [0, 1], shape (B,).
        """
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=device)
        else:
            times = torch.rand(batch_size, device=device)
        return times

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------
    def get_diffusion_loss(
        self,
        gamma_t: Tensor,
        times: Tensor,
        pred_noise: Tensor,
        noise: Tensor,
        bpd_factor: float,
    ) -> Tensor:
        """Compute diffusion loss term (Eq. 17 in arxiv:2107.00630)."""
        gamma_grad = autograd.grad(
            gamma_t,
            times,
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]  # shape: (B,)

        pred_loss = ((pred_noise - noise) ** 2).flatten(start_dim=1).sum(dim=-1)
        return bpd_factor * 0.5 * pred_loss * gamma_grad

    def get_latent_loss(self, x: Tensor, bpd_factor: float) -> Tensor:
        """Latent loss to ensure the prior is truly Gaussian (KL to N(0, I))."""
        gamma_1 = self.gamma(torch.tensor([1.0], device=x.device))
        sigma_1_sq = torch.sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x**2
        return bpd_factor * kl_std_normal(mean_sq, sigma_1_sq).flatten(start_dim=1).sum(
            dim=-1
        )

    def get_reconstruction_loss(self, x: Tensor, bpd_factor: float) -> Tensor:
        """Measure reconstruction error at t = 0."""
        noise_0 = torch.randn_like(x)
        times = torch.tensor([0.0], device=x.device)
        z_0, gamma_0 = self.variance_preserving_map(
            x,
            times=times,
            noise=noise_0,
        )
        # Generate a sample for z_0 -> closest to the data
        alpha_0 = torch.sqrt(torch.sigmoid(-gamma_0))
        z_0_rescaled = z_0 / alpha_0

        log_prob = Normal(loc=z_0_rescaled, scale=self.data_noise).log_prob(x)
        return -bpd_factor * log_prob.flatten(start_dim=1).sum(dim=-1)

    # ------------------------------------------------------------------
    # Total VDM loss (ELBO)
    # ------------------------------------------------------------------
    def get_loss(
        self,
        x: Tensor,
        noise: Optional[Tensor] = None,
        reduction: str = "mean",
        **kwargs,
    ):
        """Compute total VDM loss (ELBO) for a batch (Eq. 11 in arxiv:2107.00630).

        Args
        ----
        x:
            Data batch x_0, shape (B, C, ...).
        noise:
            Optional pre-sampled noise ε.
        reduction:
            'mean' or 'none'.

        Extra kwargs are passed to the score_model via get_pred_noise,
        e.g., s_conditioning=..., v_conditionings=[...], etc.
        """
        # ---------- NaN/Inf 체크: 입력 x ----------
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError("[NaN DETECTED] x has NaN/Inf before VDM.get_loss")

        # Optional classifier-free guidance mask on vector conditionings
        if self.p_cfg is not None:
            assert "v_conditionings" in kwargs, "Need v_conditionings to mask out"
            batch_size = x.shape[0]
            mask = torch.rand(batch_size, device=x.device) < self.p_cfg
            v_conditionings = kwargs["v_conditionings"]
            for v in v_conditionings:
                v[mask, :] = -1.0
            kwargs["v_conditionings"] = v_conditionings

        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))

        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(x.shape[0], device=x.device).requires_grad_(True)

        if noise is None:
            noise = torch.randn_like(x)

        x_t, gamma_t = self.variance_preserving_map(x=x, times=times, noise=noise)

        # ---------- NaN/Inf 체크: 중간 텐서들 ----------
        for name, tensor in [
            ("times", times),
            ("x_t", x_t),
            ("gamma_t", gamma_t),
            ("noise", noise),
        ]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise RuntimeError(f"[NaN DETECTED] {name} has NaN/Inf in VDM.get_loss")

        # Predict noise added
        pred_noise = self.get_pred_noise(
            zt=x_t,
            gamma_t=gamma_t.squeeze(),
            **kwargs,
        )

        if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
            raise RuntimeError(
                "[NaN DETECTED] pred_noise (score_model output) has NaN/Inf"
            )

        # Diffusion loss
        diffusion_loss = self.get_diffusion_loss(
            gamma_t=gamma_t,
            times=times,
            pred_noise=pred_noise,
            noise=noise,
            bpd_factor=bpd_factor,
        )

        if torch.isnan(diffusion_loss).any() or torch.isinf(diffusion_loss).any():
            raise RuntimeError("[NaN DETECTED] diffusion_loss has NaN/Inf")

        # Latent loss: KL[q(z_1 | x) || N(0, I)]
        latent_loss = self.get_latent_loss(
            x=x,
            bpd_factor=bpd_factor,
        )

        if torch.isnan(latent_loss).any() or torch.isinf(latent_loss).any():
            raise RuntimeError("[NaN DETECTED] latent_loss has NaN/Inf")

        # Reconstruction loss: -E_{q(z_0 | x)} [log p(x | z_0)]
        recons_loss = self.get_reconstruction_loss(
            x=x,
            bpd_factor=bpd_factor,
        )

        if torch.isnan(recons_loss).any() or torch.isinf(recons_loss).any():
            raise RuntimeError("[NaN DETECTED] recons_loss has NaN/Inf")

        # Overall loss, shape (B,).
        loss = diffusion_loss + latent_loss + recons_loss

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise RuntimeError("[NaN DETECTED] total loss has NaN/Inf")

        if reduction == "mean":
            metrics = {
                "elbo": loss.mean(),
                "diffusion_loss": diffusion_loss.mean(),
                "latent_loss": latent_loss.mean(),
                "reconstruction_loss": recons_loss.mean(),
            }
            return loss.mean(), metrics

        return loss, {
            "elbo": loss,
            "diffusion_loss": diffusion_loss,
            "latent_loss": latent_loss,
            "reconstruction_loss": recons_loss,
        }


    # ------------------------------------------------------------------
    # Alpha / Sigma
    # ------------------------------------------------------------------
    def alpha(self, gamma_t: Tensor) -> Tensor:
        """Compute alpha(t) (Eq. 4, arxiv:2107.00630)."""
        sig = torch.sigmoid(-gamma_t)
        sig = torch.clamp(sig, min=1e-12, max=1.0)  # 안정화
        return torch.sqrt(sig)

    def sigma(self, gamma_t: Tensor) -> Tensor:
        """Compute sigma(t) (Eq. 3, arxiv:2107.00630)."""
        sig = torch.sigmoid(gamma_t)
        sig = torch.clamp(sig, min=1e-12, max=1.0)
        return torch.sqrt(sig)


    # ------------------------------------------------------------------
    # Linking to score model
    # ------------------------------------------------------------------
    def get_pred_noise(self, zt: Tensor, gamma_t: Tensor, **kwargs) -> Tensor:
        """Get predicted noise from the score model.

        VDM passes a normalized time t_norm in [0, 1] to the score network.
        """
        # Normalized time in [0, 1]
        t_norm = (gamma_t - self.gamma_min) / (self.gamma_max - self.gamma_min)

        # Standard case: no classifier-free guidance at sampling,
        # or during training (self.training=True).
        if self.w_cfg is None or self.training:
            return self.score_model(
                zt,
                t=t_norm,
                **kwargs,
            )

        # Classifier-free guidance at sampling time
        assert "v_conditionings" in kwargs, "Need v_conditionings to mask out"
        v_conditionings = kwargs["v_conditionings"]

        # Unconditional branch: mask out all conditioning
        v_conditionings_uncond = [v.clone() for v in v_conditionings]
        for v in v_conditionings_uncond:
            v[:] = -1.0

        kwargs["v_conditionings"] = v_conditionings_uncond
        pred_noise_uncond = self.score_model(
            zt,
            t=t_norm,
            **kwargs,
        )

        # Conditional branch: original conditioning
        kwargs["v_conditionings"] = v_conditionings
        pred_noise_cond = self.score_model(
            zt,
            t=t_norm,
            **kwargs,
        )

        # Guidance combination
        return pred_noise_uncond + self.w_cfg * (pred_noise_cond - pred_noise_uncond)

    # ------------------------------------------------------------------
    # Reverse sampling
    # ------------------------------------------------------------------
    def sample_zs_given_zt(
        self,
        zt: Tensor,
        t: Tensor,
        s: Tensor,
        return_ddnm: bool = False,
        **kwargs,
    ):
        """Sample p(z_s | z_t, x) used for standard ancestral sampling (Eq. 34)."""
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = self.alpha(gamma_t)
        alpha_s = self.alpha(gamma_s)
        sigma_t = self.sigma(gamma_t)
        sigma_s = self.sigma(gamma_s)

        pred_noise = self.get_pred_noise(
            zt=zt,
            gamma_t=gamma_t,
            **kwargs,
        )

        if not return_ddnm:
            mean = alpha_s / alpha_t * (zt - c * sigma_t * pred_noise)
            scale = sigma_s * torch.sqrt(c)
            return mean + scale * torch.randn_like(zt)

        # DDNM branch
        gamma_0 = self.gamma(torch.tensor([0.0], device=zt.device))
        alpha_0 = self.alpha(gamma_0)
        sigma_0 = self.sigma(gamma_0)
        c0 = -expm1(gamma_0 - gamma_t)
        x_0t = alpha_0 / alpha_t * (zt - c0 * sigma_t * pred_noise)

        alpha_ts = alpha_t / alpha_s
        sigma_ts_sq = sigma_t**2 - (alpha_ts**2) * (sigma_s**2)
        w_z = alpha_ts * (sigma_s / sigma_t) ** 2
        w_x_0t = alpha_s * sigma_ts_sq / (sigma_t**2)
        scale = torch.sqrt(sigma_ts_sq * (sigma_s / sigma_t) ** 2)

        return w_z, w_x_0t, x_0t, scale

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        n_sampling_steps: int,
        device: str,
        z: Optional[Tensor] = None,
        return_all: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Tensor:
        """Generate new samples given (optional) conditioning vectors."""
        if z is None:
            z = torch.randn(
                (batch_size, *self.score_model.shape),
                device=device,
            )

        steps = torch.linspace(
            1.0,
            0.0,
            n_sampling_steps + 1,
            device=device,
        )

        if return_all:
            zs = []

        iterator = (
            trange(n_sampling_steps, desc="sampling")
            if verbose
            else range(n_sampling_steps)
        )

        for i in iterator:
            z = self.sample_zs_given_zt(
                zt=z,
                t=steps[i],
                s=steps[i + 1],
                **kwargs,
            )
            if return_all:
                zs.append(z)

        if return_all:
            return torch.stack(zs, dim=0)
        return z


# =========================================================================
# CUNet: Conditional U-Net backbone (from MLtools, adapted)
# =========================================================================


class CUNet(nn.Module):
    """
    CUNet backbone for diffusion / score models.

    Notes for 3D VDM use-case:
    --------------------------
    - For unconditional VDM:
        * v_conditioning_dims = []
        * Call forward(z_t, t=t_norm, s_conditioning=x_cond) where:
            z_t: noisy target (e.g. rho / tscphi), shape (B, 1, D, H, W)
            s_conditioning: galaxy-based conditioning (e.g. [ngal, vpec]),
                            shape (B, 2, D, H, W)
    - For 3D fields:
        * shape must be (C, D, H, W)
        * mid_attn should generally be False.
    """

    def __init__(
        self,
        shape=(1, 256, 256),
        out_channels=None,
        chs=[48, 96, 192, 384],
        s_conditioning_channels: int = 0,
        v_conditioning_dims: list = [],
        v_conditioning_type: str = "common_zerolinear",
        v_embedding_dim: int = 64,
        v_augment: bool = False,
        v_embed_no_s_gelu: bool = False,
        t_conditioning: bool = False,
        t_embedding_dim: int = 64,
        init_scale: float = 0.02,
        num_res_blocks: int = 1,
        norm_groups: int = 8,
        mid_attn: bool = True,
        n_attention_heads: int = 4,
        dropout_prob: float = 0.1,
        conv_padding_mode: str = "zeros",
        verbose: int = 0,
    ):
        super().__init__()
        self.shape = shape  # e.g. (C, D, H, W) or (C, H, W)
        self.chs = chs
        self.dim = len(self.shape) - 1  # 2D or 3D
        self.in_channels = self.shape[0]

        if out_channels is None:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels

        self.s_conditioning_channels = s_conditioning_channels
        self.v_conditioning_dims = v_conditioning_dims
        self.v_conditioning_type = v_conditioning_type
        self.common, self.cond_proj_type = v_conditioning_type.split("_")
        self.common = self.common == "common"
        self.v_embedding_dim = v_embedding_dim
        self.v_augment = v_augment
        if self.v_augment:
            assert self.common
        self.v_embed_no_s_gelu = v_embed_no_s_gelu

        self.t_conditioning = t_conditioning
        self.t_embedding_dim = t_embedding_dim
        self.norm_groups = norm_groups
        self.mid_attn = mid_attn
        if self.mid_attn and self.dim == 3:
            # In practice, for 3D you probably want mid_attn=False
            raise ValueError("3D attention very highly discouraged.")
        self.n_attention_heads = n_attention_heads
        self.dropout_prob = dropout_prob
        self.verbose = verbose

        # ------------------------------------------------------------------
        # Build conditioning dims: [t_cond, v_cond1, v_cond2, ...]
        # ------------------------------------------------------------------
        conditioning_dims = []

        # time conditioning
        if self.t_conditioning:
            self.t_conditioning_dim = int(4 * self.t_embedding_dim)
            self.embed_t_conditioning = nn.Sequential(
                nn.Linear(self.t_embedding_dim, self.t_conditioning_dim),
                nn.GELU(),
                nn.Linear(self.t_conditioning_dim, self.t_conditioning_dim),
                nn.GELU(),
            )
            conditioning_dims.append(self.t_conditioning_dim)

        # vector conditionings
        if len(self.v_conditioning_dims) > 0:
            self.embeds_v_conditionings = nn.ModuleList()
            for v_conditioning_dim in self.v_conditioning_dims:
                if self.common:
                    dim_mlp = (
                        2 * self.v_embedding_dim
                        if self.v_augment
                        else self.v_embedding_dim
                    )
                    self.embeds_v_conditionings.append(
                        nn.Sequential(
                            nn.Linear(v_conditioning_dim, dim_mlp),
                            nn.GELU(),
                            zero_init(nn.Linear(dim_mlp, dim_mlp))
                            if self.v_augment
                            else nn.Linear(dim_mlp, dim_mlp),
                            nn.GELU() if not self.v_embed_no_s_gelu else nn.Identity(),
                        )
                    )
                    conditioning_dims.append(self.v_embedding_dim)
                else:
                    self.embeds_v_conditionings.append(nn.Identity())
                    conditioning_dims.append(v_conditioning_dim)

        if len(conditioning_dims) == 0:
            conditioning_dims = None
        self.conditioning_dims = conditioning_dims

        # ------------------------------------------------------------------
        # Core UNet-like architecture
        # ------------------------------------------------------------------
        self.conv_kernel_size = 3
        self.norm_eps = 1e-6
        self.norm_affine = True
        self.act = "gelu"
        self.num_res_blocks = num_res_blocks
        assert self.conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"

        norm_params = dict(
            num_groups=self.norm_groups, eps=self.norm_eps, affine=self.norm_affine
        )
        assert self.act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"

        def get_act():
            if self.act == "gelu":
                return nn.GELU()
            elif self.act == "relu":
                return nn.ReLU()
            elif self.act == "silu":
                return nn.SiLU()

        padding = self.conv_kernel_size // 2
        conv_params = dict(
            kernel_size=self.conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        nca_params = dict(
            norm_params=norm_params, get_act=get_act, conv_params=conv_params
        )
        resnet_params = dict(
            dim=self.dim,
            conditioning_dims=self.conditioning_dims,
            dropout_prob=self.dropout_prob,
            nca_params=nca_params,
            cond_proj_type=self.cond_proj_type,
        )

        self.n_sizes = len(self.chs)
        self.conv_in = get_conv(
            self.in_channels + self.s_conditioning_channels,
            self.chs[0],
            dim=self.dim,
            **conv_params,
        )

        # down path
        self.downs = nn.ModuleList()
        for i_level in range(self.n_sizes):
            ch_in = self.chs[0] if i_level == 0 else self.chs[i_level - 1]
            ch_out = self.chs[i_level]
            resnets = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in, ch_out, **resnet_params))
                ch_in = ch_out
            down = ResNetDown(resnets)
            self.downs.append(down)

        # middle
        self.mid1 = ResNetBlock(ch_out, ch_out, **resnet_params)
        if self.mid_attn:
            self.mid_attn1 = AttnBlock(
                ch_out,
                n_heads=self.n_attention_heads,
                dim=self.dim,
                norm_params=norm_params,
            )
        self.mid2 = ResNetBlock(ch_out, ch_out, **resnet_params)

        # upsampling
        self.ups = nn.ModuleList()
        ch_skip = 0
        for i_level in reversed(range(self.n_sizes)):
            ch_in = self.chs[i_level]
            ch_out = self.chs[0] if i_level == 0 else self.chs[i_level - 1]
            resnets = nn.ModuleList()
            for i_resnet in range(self.num_res_blocks):
                resnets.append(
                    ResNetBlock(
                        ch_in + (ch_skip if i_resnet == 0 else 0),
                        ch_in,
                        **resnet_params,
                    )
                )
            up = ResNetUp(resnet_blocks=resnets, ch_out=ch_out)
            ch_skip = ch_out
            self.ups.append(up)

        self.norm_out = nn.GroupNorm(num_channels=ch_out, **norm_params)
        self.act_out = get_act()
        self.conv_out = get_conv(
            in_channels=ch_out,
            out_channels=self.out_channels,
            dim=self.dim,
            init=zero_init,
            **conv_params,
        )

        if self.in_channels != self.out_channels:
            self.conv_residual_out = get_conv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                dim=self.dim,
                init=zero_init,
                **conv_params,
            )

        # small init for stability
        for _, p in self.named_parameters():
            p.data *= init_scale

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
        s_conditioning: Optional[Tensor] = None,
        v_conditionings: Optional[list] = None,
    ) -> Tensor:
        """
        x:
            Input tensor, typically the noisy target z_t, shape (B, C, ...).
        t:
            Normalized time in [0, 1], shape (B,). Required if t_conditioning=True.
        s_conditioning:
            Spatial conditioning (e.g. [ngal, vpec]), shape (B, C_s, ...).
        v_conditionings:
            List of vector conditionings (e.g. cosmological parameters).
        """
        # spatial (image/field) conditioning
        if s_conditioning is not None:
            if self.s_conditioning_channels != s_conditioning.shape[1]:
                raise ValueError(
                    f"Expected s_conditioning to have {self.s_conditioning_channels} "
                    f"channels, but got {s_conditioning.shape[1]}"
                )
            x_concat = torch.cat((x, s_conditioning), dim=1)
        else:
            x_concat = x

        conditionings = []

        # time conditioning
        if t is not None:
            if not self.t_conditioning:
                raise ValueError("t is not None, but t_conditioning is False")
            t = t.expand(x_concat.shape[0]).clone()
            assert t.shape == (x_concat.shape[0],)

            t_embedding = get_timestep_embedding(t, self.t_embedding_dim)
            t_cond = self.embed_t_conditioning(t_embedding)
            conditionings.append(t_cond)
        else:
            if self.t_conditioning:
                raise ValueError("t is None, but t_conditioning is True")

        # vector conditionings (if used)
        if v_conditionings is not None:
            if len(v_conditionings) != len(self.v_conditioning_dims):
                raise ValueError(
                    f"Expected {len(self.v_conditioning_dims)} v_conditionings, "
                    f"but got {len(v_conditionings)}"
                )
            for i, v_conditioning in enumerate(v_conditionings):
                if v_conditioning.shape[1] != self.v_conditioning_dims[i]:
                    raise ValueError(
                        f"Expected v_conditioning to have {self.v_conditioning_dims[i]} "
                        f"channels, but got {v_conditioning.shape[1]}"
                    )
                v_cond = self.embeds_v_conditionings[i](v_conditioning)
                if self.v_augment:
                    means = v_cond[:, ::2]
                    stds = torch.exp(v_cond[:, 1::2])
                    v_cond = means + stds * torch.randn_like(stds)
                conditionings.append(v_cond)

        if len(conditionings) == 0:
            conditionings = None

        # U-Net forward
        h = self.conv_in(x_concat)
        skips = []
        for i, down in enumerate(self.downs):
            h, h_skip = down(
                h,
                conditionings=conditionings,
                no_down=(i == (len(self.downs) - 1)),
            )
            if h_skip is not None:
                skips.append(h_skip)

        # middle
        h = self.mid1(h, conditionings=conditionings)
        if self.mid_attn:
            h = self.mid_attn1(h)
        h = self.mid2(h, conditionings=conditionings)

        # upsampling
        for i, up in enumerate(self.ups):
            x_skip = skips.pop() if len(skips) > 0 else None
            h = up(
                h,
                x_skip=x_skip,
                conditionings=conditionings,
                no_up=(i == self.n_sizes - 1),
            )

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        if self.in_channels != self.out_channels:
            x = self.conv_residual_out(x)

        return h + x
