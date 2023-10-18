import gc
import math
from functools import partial

import h5py
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import Adam
from prepare_data import simulate_new, simulate_on_grid, CryoEMDataset, normalize_map
from util.map_splitter import split_map, reconstruct_map
from pathlib import Path
from util.density_map import DensityMap
import tempfile
import os
import shutil
import wandb
from torch.cuda.amp import GradScaler, autocast
import random
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from util.resample import resample
from torch.optim.lr_scheduler import MultiplicativeLR
import json


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv3d(dim, default(dim_out, dim), kernel_size=(3, 3, 3), padding=1)
    )


def Downsample(dim, dim_out=None):
    """
    Old version: dim * 4 * 4 * 4 * dim = 64 * dim^2
    return nn.Conv3d(dim, dim, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=1)
    New version: dim * 8 * 1 * 1 * 1 * dim = 8 * dim^2
    Equivalent to:
    return nn.Conv3d(dim, dim, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=1)
    """
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) (d p3) -> b (c p1 p2 p3) h w d', p1=2, p2=2, p3=2),
        nn.Conv3d(dim * 8, default(dim_out, dim), kernel_size=(1, 1, 1))
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, kernel_size=(3, 3, 3), padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.acti = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.acti(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, emb_dim=None, groups=8, dropout=False):
        super().__init__()
        assert emb_dim, "embedding size be passed in"
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, dim_out * 2)
        ) if exists(emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.dropout = nn.Dropout(0.1) if dropout else None
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        if exists(self.dropout):
            h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=2, norm=True):
        super().__init__()

        self.t_mlp = (nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim)) if emb_dim else None)
        self.in_A_mlp = (nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim)) if emb_dim else None)
        self.out_A_mlp = (nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim)) if emb_dim else None)

        self.ds_conv = nn.Conv3d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv3d(dim, dim_out * mult, 3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv3d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, input_emb=None, output_emb=None):
        h = self.ds_conv(x)

        if exists(self.t_mlp) and exists(self.in_A_mlp) and exists(self.out_A_mlp):
            assert exists(time_emb) and exists(input_emb) and exists(output_emb), "3 embeddings must be passed in"
            t = rearrange(self.t_mlp(time_emb), "b c -> b c 1 1 1")
            i = rearrange(self.in_A_mlp(input_emb), "b c -> b c 1 1 1")
            o = rearrange(self.out_A_mlp(output_emb), "b c -> b c 1 1 1")
            h = h + t + i + o

        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv)
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=h, y=w, z=d)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        out = rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=h, y=w, z=d)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    """
    Do note
    """

    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mult=(1, 2, 4, 8),
            channels=3,
            resnet_block_groups=8,
            use_ConvNext=False,
            ConvNext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(channels * 2, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_ConvNext:
            block_klass = partial(ConvNextBlock, mult=ConvNext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        def generate_embedding(dim):
            time_dim = dim * 4
            return nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

        embedding_dim = dim * 4
        # time embeddings
        self.time_mlp = generate_embedding(dim)
        # input resolution embeddings
        self.input_A_mlp = generate_embedding(dim)
        # output resolution embeddings
        self.output_A_mlp = generate_embedding(dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, emb_dim=embedding_dim),
                        block_klass(dim_out, dim_out, emb_dim=embedding_dim),
                        Residual(PreNorm(dim_out, Attention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, emb_dim=embedding_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, emb_dim=embedding_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, emb_dim=embedding_dim),
                        block_klass(dim_in, dim_in, emb_dim=embedding_dim),
                        Residual(PreNorm(dim_in, Attention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv3d(dim, out_dim, 1)
        )

    def forward(self, x, time, in_resolution, out_resolution):
        x = self.init_conv(x)

        t = self.time_mlp(time)
        i = self.input_A_mlp(in_resolution)
        o = self.output_A_mlp(out_resolution)

        h = []

        # down sample
        for block1, block2, attn, downSample in self.downs:
            x = block1(x, t, i, o)
            x = block2(x, t, i, o)
            x = attn(x)
            h.append(x)
            x = downSample(x)

        # bottleneck
        x = self.mid_block1(x, t, i, o)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, i, o)

        # up sample
        for block1, block2, attn, upSample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, i, o)
            x = block2(x, t, i, o)
            x = attn(x)
            x = upSample(x)

        return self.final_conv(x)


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original DDPM paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule from improved DDPM:
    https://arxiv.org/abs/2102.09672
    https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule:
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule from improved DDPM:
    https://arxiv.org/abs/2102.09672
    https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule:
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_noise',
            beta_schedule='cosine',
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.,
            auto_normalize=True,
            min_snr_loss_weight=False,
            min_snr_gamma=5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0',
                             'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def condition_mean(self, cond_fn, mean, variance, x, t, guidance_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **guidance_kwargs)
        new_mean = (
                mean.float() + variance * gradient.float()
        )
        print("gradient: ", (variance * gradient.float()).mean())
        return new_mean

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, cond_fn=None, guidance_kwargs=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )
        if exists(cond_fn) and exists(guidance_kwargs):
            model_mean = self.condition_mean(cond_fn, model_mean, variance, x, batched_times, guidance_kwargs)

        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False, cond_fn=None, guidance_kwargs=None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, cond_fn, guidance_kwargs)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps=False, cond_fn=None, guidance_kwargs=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=True)

            imgs.append(img)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False, cond_fn=None, guidance_kwargs=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps=return_all_timesteps,
                         cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


def set_seeds(seed):
    """Sets all seeds and disables non-deset_seedsterminism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_model(args):
    model = Unet(
        dim=args.hide_dim,
        channels=args.channels,
        dim_mult=(1, 2, 4, 8)
    )
    if args.load:
        path = os.path.join(args.model_dir, args.load)
        model.load_state_dict(torch.load(path))
        print("Loaded model from", path)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(args.device)
    return model


def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()


def train(args, model, trainloader, valloader=None):
    scaler = GradScaler()
    optimizer = Adam(model.parameters(), lr=args.lr)
    # Decrease learning rate by x time for every epoch
    multi = 1 / 2
    decay = multi ** (1 / 500)
    # decay = 1.0
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda x: decay)
    it = 0
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    saving = False
    for epoch in range(1, args.epochs):
        for in_batch, in_resolution, out_batch, out_resolution in tqdm(trainloader, desc="Train"):
            it += 1
            optimizer.zero_grad()
            batch_size = len(in_batch)
            with autocast():
                in_batch = in_batch.to(args.device)
                in_resolution = in_resolution.to(args.device)
                out_batch = out_batch.to(args.device)
                out_resolution = out_resolution.to(args.device)

                # Algorithm 1 line 3: sample t uniformly for every example in the batch
                t_tensor = torch.randint(0, 1000, (batch_size,), device=args.device).long()

                loss = p_losses(
                    model,
                    in_batch.float(),
                    out_batch.float(),
                    t_tensor,
                    in_resolution,
                    out_resolution,
                    loss_type="huber"
                )

            wandb.log({"Train_Loss": loss.item(), 'epoch': epoch, 'lr': scheduler.get_last_lr()[0]}, step=it)
            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                saving = True

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # if it % 1000 == 0 and saving:
            #     weights = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            #     torch.save(weights, os.path.join(args.model_dir, f'model_dim_{args.hide_dim}_{it // 1000}.pth'))
            #     saving = False

        # optimizer.zero_grad()
        # if valloader is not None:
        #     total_loss = []
        #     for in_batch, in_resolution, out_batch, out_resolution in tqdm(valloader, desc="Validation"):
        #         optimizer.zero_grad()
        #         batch_size = len(in_batch)
        #         with autocast():
        #             in_batch = in_batch.to(args.device)
        #             in_resolution = in_resolution.to(args.device)
        #             out_batch = out_batch.to(args.device)
        #             out_resolution = out_resolution.to(args.device)
        #             t_tensor = torch.randint(0, timesteps, (batch_size,), device=args.device).long()
        #
        #             loss = p_losses(
        #                 model,
        #                 in_batch,
        #                 out_batch,
        #                 t_tensor,
        #                 in_resolution,
        #                 out_resolution,
        #                 loss_type="huber",
        #                 reduction="none"
        #             )
        #             loss = torch.mean(loss, (1, 2, 3, 4))
        #             total_loss += list(loss.detach().cpu())
        #     val_loss = sum(total_loss) / len(total_loss)
        #     wandb.log({"Validation_Loss": val_loss}, step=it)
        #     wandb.log({'Validation_Loss_histogram': wandb.Histogram(total_loss)}, step=it)
        #
        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         weights = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        #         torch.save(weights, os.path.join(args.model_dir, f'model_dim_{args.hide_dim}_best.pth'))
        #
        # if args.save_last_model and epoch % 1000 == 0:
        #     weights = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        #     torch.save(weights, os.path.join(args.model_dir, f'model_dim_{args.hide_dim}_last.pth'))


@torch.no_grad()
def p_sample(args, model, in_batch, x, t_tensor, t, in_label, out_label):
    betas_t = extract(betas, t_tensor, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_tensor, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    in_tensor = torch.cat((in_batch, x), dim=1)
    predicted_noise = model(in_tensor, t_tensor, in_label, out_label)
    model_mean = sqrt_recip_alphas_t * (x - (betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t))

    if t == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t_tensor, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(args, model, in_batch):
    b = in_batch.shape[0]
    # start from pure noise (for each example in the batch)
    x = torch.randn(in_batch.shape, device=args.device).float()
    # imgs = []
    in_resolution = torch.full((b,), args.input_resolution, device=args.device, dtype=torch.float)
    out_resolution = torch.full((b,), args.output_resolution, device=args.device, dtype=torch.float)

    for t in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t_tensor = torch.full((b,), t, device=args.device, dtype=torch.long)
        x = p_sample(args, model, in_batch, x, t_tensor, t, in_resolution, out_resolution)
        # imgs.append(img.cpu().numpy())
    return x


@torch.no_grad()
def predict(args, model, density_map):
    in_map = np.stack(split_map(density_map.data, box_size=args.image_size, core_size=args.core_size))
    out_map = []
    for i in range(0, -(-len(in_map) // args.batch_size)):
        print("Batch:", i + 1, "Total:", -(-len(in_map) // args.batch_size))
        in_map_batch = torch.from_numpy(in_map[(i * args.batch_size):((i + 1) * args.batch_size)]).to(args.device)
        out_map_batch = p_sample_loop(args, model, in_map_batch.float())
        for out_map_i in out_map_batch:
            out_map.append(out_map_i[0].detach().cpu().numpy())
    out_map = reconstruct_map(out_map, target_shape=density_map.shape, box_size=args.image_size,
                              core_size=args.core_size)
    density_map.data = out_map
    return density_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--core_size', type=int, default=50, help='core size')
    parser.add_argument('--hide_dim', type=int, default=64, help='batch size')
    parser.add_argument('--channels', type=int, default=1, help='channels')
    parser.add_argument('--batch_size', type=int, default=18, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs')
    parser.add_argument('--dataset_dir', type=str, default='/data/sbcaesar/SR-CryoEM-testset')
    parser.add_argument('--load', type=str, help='load model name')
    parser.add_argument('--model_dir', type=str, default='model', help='model dir')
    parser.add_argument('--save_last_model', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--multi_gpu', action='store_true', help='muti_gpu')
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--predict', action='store_true', help='predict')
    parser.add_argument('--input_path', type=str, default='/data/sbcaesar/SR-CryoEM-testset/EMD-4054/emd_4054.map',
                        help='input path')
    parser.add_argument('--input_resolution', type=float, default=5.9, help='input resolution')
    parser.add_argument('--output_resolution', type=float, default=3.0, help='output resolution')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    args = parser.parse_args()
    args.input_path = '/data/sbcaesar/SR-CryoEM-testset/EMD-22287/emd_22287.map'

    if args.device == 'cuda':
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seeds(args.seed)

    model = get_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    if args.train:
        master_name = 'continue small dataset with lr decay'
        WANDB_RUN_NAME = f'lr_{args.lr}_batch_size_{args.batch_size}_hide_dim_{args.hide_dim}'
        wandb.init(project='SR_CryoEM', name=master_name + WANDB_RUN_NAME, config=args, save_code=True)
        print("Super Resolution Training Starting...")
        print("Training LR:", args.lr)

        hdf5 = h5py.File(os.path.join(args.dataset_dir, 'SR_CryoEM_Dataset.hdf5'), 'r')

        train_set = CryoEMDataset(hdf5['train'])
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        # val_set = CryoEMDataset(hdf5['val'])
        # valloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

        train(args, model, trainloader)

    if args.predict:
        print("Super Resolution Inference Starting...")
        print("Input map path:", args.input_path)
        assert args.input_resolution > args.output_resolution, "Input resolution must be larger than output resolution"
        print("Super resolution from", args.input_resolution, "to", args.output_resolution)
        density_map = DensityMap.open(args.input_path)
        density_map = resample(density_map, voxel_size=1.0, contour=0.01)
        density_map = normalize_map(density_map)
        density_map = predict(args, model, density_map)
        percentile = np.percentile(density_map.data[np.nonzero(density_map.data)], 0.1)
        density_map.data += percentile
        output_path = args.input_path[:-4] + f'_sr_{args.output_resolution}.mrc'
        density_map.save(output_path)
        print("Output map path:", output_path)
