import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import Adam
from prepare_data import simulate_new, simulate_on_grid, CryoEMDataset
from util.map_splitter import split_map, reconstruct_map
from util.normalize import normalize_map
from pathlib import Path
from util.density_map import DensityMap
import tempfile
import os
import shutil
import wandb


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv3d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, emb_dim=None, groups=8):
        super().__init__()
        assert emb_dim, "embedding size be passed in"
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(emb_dim, dim))
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=2, norm=True):
        super().__init__()

        self.t_mlp = (nn.Sequential(nn.GELU(), nn.Linear(emb_dim, dim)) if emb_dim else None)
        self.in_A_mlp = (nn.Sequential(nn.GELU(), nn.Linear(emb_dim, dim)) if emb_dim else None)
        self.out_A_mlp = (nn.Sequential(nn.GELU(), nn.Linear(emb_dim, dim)) if emb_dim else None)

        self.ds_conv = nn.Conv3d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv3d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
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
        b, c, h, w, p = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=h, y=w, z=p)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w, p = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=h, y=w, z=p)
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
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv3d(channels * 2, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        def generate_embedding(dim):
            time_dim = dim * 4
            return nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
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
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
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
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
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


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = 200

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

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


def p_losses(model, x, y, t, in_resolution, out_resolution, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(y)

    y_noisy = q_sample(y=y, t=t, noise=noise)
    x_y_noisy = torch.cat((x, y_noisy), dim=1)
    predicted_noise = model(x_y_noisy, t, in_resolution, out_resolution)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def get_model(args):
    model = Unet(
        dim=args.image_size,
        channels=args.channels,
        dim_mults=(1, 2, 4,)
    )
    if args.load:
        model.load_state_dict(torch.load("model/model_1.pth"))
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(args.device)
    return model


def get_dataset(args):
    if os.path.isfile("/data/sbcaesar/Mac-SR-CryoEM/dataset/3186_dataset.pt"):
        dataset = torch.load("/data/sbcaesar/Mac-SR-CryoEM/dataset/3186_dataset.pt")
    else:
        in_map_list = list()
        out_map_list = list()
        in_label = list()
        out_label = list()

        # Adding simulated data
        pdb = '/data/sbcaesar/Mac-SR-CryoEM/dataset/5fj6.pdb'
        for in_res in np.arange(4.0, 10.0, 0.4):
            in_map = simulate_new(pdb, in_res, 1)
            in_map = normalize_map(in_map)
            in_data = split_map(in_map.data)
            tmp_dir = tempfile.mkdtemp(prefix='deeptracer_preprocessing')
            tmp_map_path = os.path.join(tmp_dir, 'in.mrc')
            in_map.save(tmp_map_path)
            for out_res in np.arange(2.5, in_res, 0.5):
                out_map = simulate_on_grid(pdb, out_res, tmp_map_path)
                out_map = normalize_map(out_map)
                out_data = split_map(out_map.data)
                in_map_list += in_data
                out_map_list += out_data
                in_label.append(torch.full((len(in_data),), in_res))
                out_label.append(torch.full((len(out_data),), out_res))
            shutil.rmtree(tmp_dir)

        # Adding real data
        in_res = 8.0
        tmp_map_path = '/data/sbcaesar/Mac-SR-CryoEM/dataset/emd_3186.map'
        in_map = DensityMap.open(tmp_map_path)
        in_map = normalize_map(in_map)
        in_data = split_map(in_map.data)
        for out_res in np.arange(2.5, in_res, 0.5):
            out_map = simulate_on_grid(pdb, out_res, tmp_map_path)
            out_map = normalize_map(out_map)
            out_data = split_map(out_map.data)
            in_map_list += in_data
            out_map_list += out_data
            in_label.append(torch.full((len(in_data),), in_res))
            out_label.append(torch.full((len(out_data),), out_res))

        dataset = CryoEMDataset(in_map_list, out_map_list, torch.cat(in_label, 0), torch.cat(out_label, 0))
        torch.save(dataset, "/data/sbcaesar/Mac-SR-CryoEM/dataset/3186_dataset.pt")
        print("Saved dataset to /data/sbcaesar/Mac-SR-CryoEM/dataset/3186_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def train(args, model, dataloader):
    optimizer = Adam(model.parameters(), lr=args.lr)
    i = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            i += 1
            optimizer.zero_grad()

            batch_size = len(batch[0])
            in_batch = batch[0].to(args.device)
            in_resolution = batch[1].to(args.device)
            out_batch = batch[2].to(args.device)
            out_resolution = batch[3].to(args.device)

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

            if step % 10 == 0:
                print("Epoch:", epoch, "Loss:", loss.item())
                wandb.log({"Loss": loss.item()}, step=i)

            loss.backward()
            optimizer.step()

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), args.model_path.format(epoch))
        else:
            torch.save(model.state_dict(), args.model_path.format(epoch))
        print("Model saved to", args.model_path.format(epoch))


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
    in_resolution = torch.full((b,), args.input_resolution, device=args.device, dtype=torch.long)
    out_resolution = torch.full((b,), args.output_resolution, device=args.device, dtype=torch.long)

    for t in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t_tensor = torch.full((b,), t, device=args.device, dtype=torch.long)
        x = p_sample(args, model, in_batch, x, t_tensor, t, in_resolution, out_resolution)
        # imgs.append(img.cpu().numpy())
    return x


@torch.no_grad()
def predict(args, model, density_map):
    in_map = np.stack(split_map(density_map.data, box_size=args.image_size, core_size=args.core_size))
    out_map = []
    for i in range(0, len(in_map), args.batch_size):
        print("Batch:", i + 1, "Total:", -(-len(in_map) // args.batch_size))
        in_map_batch = torch.from_numpy(in_map[i:i + args.batch_size]).to(args.device)
        out_map_batch = p_sample_loop(args, model, in_map_batch.float())
        for out_map_i in out_map_batch:
            out_map.append(out_map_i[0].detach().cpu().numpy())
    out_map = reconstruct_map(out_map, target_shape=density_map.shape, box_size=args.image_size, core_size=args.core_size)
    density_map.data = out_map
    return density_map


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--core_size', type=int, default=50, help='core size')
    parser.add_argument('--channels', type=int, default=1, help='channels')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--load', action='store_true', help='load model')
    parser.add_argument('--model_path', type=str, default='model/model_{}.pth', help='model path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--multi_gpu', default=True, action='store_true', help='muti_gpu')
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--predict', action='store_true', help='predict')
    parser.add_argument('--input_path', type=str, default='dataset/sim7.mrc', help='input path')
    parser.add_argument('--input_resolution', type=float, default=7.0, help='input resolution')
    parser.add_argument('--output_resolution', type=float, default=3.0, help='output resolution')
    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(args)

    if args.train:
        wandb.init(sync_tensorboard=False,
                   project="Diffusion-Super-Resolution",
                   job_type="CleanRepo",
                   config=args,
                   )
        print("Super Resolution Training Starting...")
        if args.load:
            print("Loaded model from", args.model_path)
        print("Training LR:", args.lr)
        dataloader = get_dataset(args)
        train(args, model, dataloader)

    if args.predict:
        print("Super Resolution Inference Starting...")
        print("Input map path:", args.input_path)
        assert args.input_resolution > args.output_resolution, "Input resolution must be larger than output resolution"
        print("Super resolution from", args.input_resolution, "to", args.output_resolution)
        density_map = DensityMap.open(args.input_path)
        predict(args, model, density_map)
        output_path = 'dataset/emd_3186_predict_{}.mrc'.format(args.output_resolution)
        density_map.save(output_path)
        print(output_path)
