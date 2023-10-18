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


timesteps = 1000

# define beta schedule
betas = cosine_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
print(timesteps, ":", alphas_cumprod)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion O(1)
def q_sample(y, t, noise=None):
    if noise is None:
        noise = torch.randn_like(y)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, y.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, y.shape
    )
    return sqrt_alphas_cumprod_t * y + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(model, x, y, t, in_resolution, out_resolution, noise=None, loss_type="l1", reduction="mean"):
    if noise is None:
        noise = torch.randn_like(y)

    y_noisy = q_sample(y=y, t=t, noise=noise)
    x_y_noisy = torch.cat((x, y_noisy), dim=1)
    predicted_noise = model(x_y_noisy, t, in_resolution, out_resolution)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise, reduction=reduction)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise, reduction=reduction)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise, reduction=reduction)
    else:
        raise NotImplementedError()

    return loss


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
        dim_mults=(1, 2, 4, 8)
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
                t_tensor = torch.randint(0, timesteps, (batch_size,), device=args.device).long()

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

            if it % 1000 == 0 and saving:
                weights = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(weights, os.path.join(args.model_dir, f'model_dim_{args.hide_dim}_{it // 1000}.pth'))
                saving = False

        optimizer.zero_grad()
        if valloader is not None:
            total_loss = []
            for in_batch, in_resolution, out_batch, out_resolution in tqdm(valloader, desc="Validation"):
                optimizer.zero_grad()
                batch_size = len(in_batch)
                with autocast():
                    in_batch = in_batch.to(args.device)
                    in_resolution = in_resolution.to(args.device)
                    out_batch = out_batch.to(args.device)
                    out_resolution = out_resolution.to(args.device)
                    t_tensor = torch.randint(0, timesteps, (batch_size,), device=args.device).long()

                    loss = p_losses(
                        model,
                        in_batch,
                        out_batch,
                        t_tensor,
                        in_resolution,
                        out_resolution,
                        loss_type="huber",
                        reduction="none"
                    )
                    loss = torch.mean(loss, (1, 2, 3, 4))
                    total_loss += list(loss.detach().cpu())
            val_loss = sum(total_loss) / len(total_loss)
            wandb.log({"Validation_Loss": val_loss}, step=it)
            wandb.log({'Validation_Loss_histogram': wandb.Histogram(total_loss)}, step=it)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                weights = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(weights, os.path.join(args.model_dir, f'model_dim_{args.hide_dim}_best.pth'))

        if args.save_last_model and epoch % 1000 == 0:
            weights = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(weights, os.path.join(args.model_dir, f'model_dim_{args.hide_dim}_last.pth'))


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
