"""
MLtools VDM + CUNet utilities (adapted from cfpark00/MLtools)

Source:
    https://github.com/cfpark00/MLtools
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_std_normal(mean_squared, var):
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)


class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class SigmoidSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.b = 1 / (np.exp(-self.gamma_min) + 1)
        self.a = 1 / (np.exp(-self.gamma_max) + 1) - self.b

    def forward(self, t):
        return -torch.log(1 / (self.a * t + self.b) - 1)


class LearnedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(gamma_min,dtype=torch.float32))
        self.w = nn.Parameter(torch.tensor(gamma_max - gamma_min,dtype=torch.float32))

    def forward(self, t):
        return self.b + self.w.abs() * t



class MonotonicLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.weight.abs(), self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class NNSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max, mid_dim=1024):
        super().__init__()
        self.mid_dim = mid_dim
        self.l1 = MonotonicLinear(1, 1, bias=True)
        self.l1.weight.data[0, 0] = gamma_max - gamma_min
        self.l1.bias.data[0] = gamma_min
        self.l2 = MonotonicLinear(1, self.mid_dim, bias=True)
        self.l3 = MonotonicLinear(self.mid_dim, 1, bias=False)

    def forward(self, t, scale=1.0):
        t_sh = t.shape
        t = t.reshape(-1, 1)
        g = self.l1(t)
        _g = 2.0 * (t - 0.5)
        _g = self.l2(_g)
        _g = 2.0 * (torch.sigmoid(_g) - 0.5)
        _g = self.l3(_g) / self.mid_dim
        _g *= scale
        g = g + _g
        return g.reshape(t_sh)

@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        torch.nn.init.zeros_(p.data)
    return module


def get_conv(in_channels, out_channels, **kwargs):
    def_params = {
        "dim": 2,
        "kernel_size": 3,
        "padding": 1,
        "stride": 1,
        "padding_mode": "zeros",
        "dilation": 1,
        "groups": 1,
        "init": lambda x: x,
        "transposed": False,
    }
    def_params.update(kwargs)
    dim = def_params.pop("dim")
    transposed = def_params.pop("transposed")
    init = def_params.pop("init")
    if dim == 2:
        conv = nn.ConvTranspose2d if transposed else nn.Conv2d
        return init(conv(in_channels, out_channels, **def_params))
    elif dim == 3:
        conv = nn.ConvTranspose3d if transposed else nn.Conv3d
        return init(conv(in_channels, out_channels, **def_params))


def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    T=1000,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    timesteps = timesteps* T
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1 or timesteps.ndim == 2
    assert embedding_dim % 2 == 0
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        base=10.0,
        device=timesteps.device,
    )
    if timesteps.ndim == 1:
        emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
        return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)
    else:
        emb = timesteps.to(dtype)[:,:,None] * inv_timescales[None,None, :] # (B, T, D/2)
        return torch.cat([emb.sin(), emb.cos()], dim=2) # (B, T, D)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads=4, dim=2, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        assert (
            self.in_channels % n_heads == 0
        ), "in_channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.dim = dim
        assert self.dim == 2 or self.dim == 3, "dim must be 2 or 3"

        norm_params = kwargs.get("norm_params", {})

        self.norm = nn.GroupNorm(num_channels=in_channels, **norm_params)

        self.q = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.k = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.v = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        if self.dim == 2:
            b, c, h, w = q.shape
            c_ = c // self.n_heads
            q = q.reshape(b, c_, self.n_heads, h * w)
            k = k.reshape(b, c_, self.n_heads, h * w)
            w_ = torch.einsum("bcnq,bcnk->bqkn", q, k)
            w_ = w_ * (int(c_) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)
            v = v.reshape(b, c_, self.n_heads, h * w)
            h_ = torch.einsum("bcnd,bqdn->bcnq", v, w_)
            h_ = h_.reshape(b, c, h, w)
            h_ = self.proj_out(h_)
        elif self.dim == 3:
            b, c, d, h, w = q.shape
            c_ = c // self.n_heads
            q = q.reshape(b, c_, self.n_heads, d * h * w)
            k = k.reshape(b, c_, self.n_heads, d * h * w)
            w_ = torch.einsum("bcnq,bcnk->bqkn", q, k)
            w_ = w_ * (int(c_) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)
            v = v.reshape(b, c_, self.n_heads, d * h * w)
            h_ = torch.einsum("bcnd,bqdn->bcnq", v, w_)
            h_ = h_.reshape(b, c, d, h, w)
            h_ = self.proj_out(h_)
        return x + h_


class ResNetBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        dim=2,
        conditioning_dims=None,
        dropout_prob=0.0,
        nca_params={},
        cond_proj_type="zerolinear"
    ):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dim = dim
        assert self.dim in [2, 3], "dim must be 2 or 3"
        self.conditioning_dims = conditioning_dims

        self.nca_params = nca_params
        norm_params = self.nca_params.get("norm_params", {})
        get_act = self.nca_params.get("get_act", lambda: nn.GELU())
        conv_params = self.nca_params.get("conv_params", {})

        self.net1 = nn.Sequential(
            nn.GroupNorm(num_channels=ch_in, **norm_params),
            get_act(),
            get_conv(ch_in, ch_out, dim=self.dim, **conv_params),
        )
        if conditioning_dims is not None:
            self.cond_projs = nn.ModuleList()
            for condition_dim in self.conditioning_dims:
                if cond_proj_type == "zerolinear":
                    self.cond_projs.append(zero_init(nn.Linear(condition_dim, ch_out)))
                elif cond_proj_type == "linear":
                    self.cond_projs.append(nn.Linear(condition_dim, ch_out))
                elif cond_proj_type == "mlp":
                    self.cond_projs.append(
                        nn.Sequential(
                            nn.Linear(condition_dim, ch_out),
                            get_act(),
                            nn.Linear(ch_out, ch_out),
                            get_act(),
                        )
                    )
                else:
                    raise ValueError(f"Unknown cond_proj_type: {cond_proj_type}")
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_channels=ch_out, **norm_params),
            get_act(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            get_conv(ch_out, ch_out, dim=self.dim, init=zero_init, **conv_params),
        )
        if ch_in != ch_out:
            self.skip_conv = get_conv(
                ch_in, ch_out, dim=self.dim, kernel_size=1, padding=0
            )

    def forward(self, x, conditionings=None):
        h = self.net1(x)
        if conditionings is not None:
            assert len(conditionings) == len(self.conditioning_dims)
            assert all(
                [
                    conditionings[i].shape == (x.shape[0], self.conditioning_dims[i])
                    for i in range(len(conditionings))
                ]
            )
            for i, conditioning in enumerate(conditionings):
                conditioning_ = self.cond_projs[i](conditioning)
                if self.dim == 2:
                    h = h + conditioning_[:, :, None, None]
                elif self.dim == 3:
                    h = h + conditioning_[:, :, None, None, None]
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        return x + h


class ResNetDown(nn.Module):
    def __init__(self, resnet_blocks, attention_blocks=None):
        super().__init__()
        self.resnet_blocks = resnet_blocks
        self.attention_blocks = attention_blocks
        self.dim = self.resnet_blocks[-1].dim
        self.down = get_conv(
            self.resnet_blocks[-1].ch_out,
            self.resnet_blocks[-1].ch_out,
            dim=self.dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x, conditionings, no_down=False):
        for i, resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x, conditionings)
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)
        if no_down:
            return x, None
        x_skip = x
        x = self.down(x)
        return x, x_skip


class ResNetUp(nn.Module):
    def __init__(
        self, resnet_blocks, attention_blocks=None, ch_out=None, conv_params={}
    ):
        super().__init__()
        self.resnet_blocks = resnet_blocks
        self.ch_out = ch_out if ch_out is not None else self.resnet_blocks[-1].ch_out
        self.attention_blocks = attention_blocks
        self.dim = self.resnet_blocks[-1].dim
        self.up = get_conv(
            self.resnet_blocks[-1].ch_out,
            self.ch_out,
            dim=self.dim,
            kernel_size=2,
            stride=2,
            padding=0,
            transposed=True,
        )

    def forward(self, x, x_skip=None, conditionings=None, no_up=False):
        for i, resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x, conditionings)
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)
        if not no_up:
            x = self.up(x)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch 2.0 doesn't support simply bias=False """

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

