from util.chimera import Chimera
import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time
import tempfile
import shutil
from util.density_map import DensityMap
from torch.utils.data import Dataset


class CryoEMDataset(Dataset):
    def __init__(self, in_map, out_map, in_label, out_label):
        self.in_map = in_map
        self.out_map = out_map
        self.in_label = in_label
        self.out_label = out_label

    def __len__(self):
        assert len(self.in_map) == len(self.in_label) == len(self.out_map) == len(self.out_label)
        return len(self.in_map)

    def __getitem__(self, idx):
        in_map = self.in_map[idx]
        out_map = self.out_map[idx]
        in_label = self.in_label[idx]
        out_label = self.out_label[idx]
        return in_map, in_label, out_map, out_label


def simulate_new(solved_structure_path: str, resolution: float, grid_size: float):
    # Chimera molmap
    tmp_dir = tempfile.mkdtemp(prefix='deeptracer_preprocessing')
    tmp_map_path = os.path.join(tmp_dir, 'in.mrc')
    Chimera.run(tmp_dir, [
        'open %s' % solved_structure_path,
        'molmap #0 %s gridSpacing %s' % (resolution, grid_size),
        'volume #0 save %s' % tmp_map_path,
    ])
    in_map = DensityMap.open(tmp_map_path)
    shutil.rmtree(tmp_dir)
    return in_map


def simulate_on_grid(solved_structure_path: str, resolution: float, map_path: str):
    # Chimera molmap
    tmp_dir = tempfile.mkdtemp(prefix='deeptracer_preprocessing')
    tmp_map_path = os.path.join(tmp_dir, 'out.mrc')
    Chimera.run(tmp_dir, [
        'open %s' % solved_structure_path,
        'open %s' % map_path,
        'volume #1 step 1',
        'molmap #0 %s onGrid #1' % resolution,
        'volume #2 save %s' % tmp_map_path,
    ])
    out_map = DensityMap.open(tmp_map_path)
    shutil.rmtree(tmp_dir)
    return out_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default='{}/Dataset/celebahq_256'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str, default='./dataset/celebahq')

    parser.add_argument('--size', type=str, default='64,512')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()
