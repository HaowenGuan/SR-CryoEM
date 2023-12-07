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
import h5py
import torch
from util.map_splitter import split_map
import random
import csv
import requests
import json
from multiprocessing import Pool, cpu_count
from util.resample import resample
import pandas as pd


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


class AutoencoderDataset(Dataset):
    def __init__(self, hdf5):
        self.cubes = hdf5['cubes']

    def __len__(self):
        return len(self.cubes)

    def __getitem__(self, idx):
        cube = torch.tensor(self.cubes[idx])
        return cube


def normalize_map(density_map: DensityMap) -> DensityMap:
    """
    Normalizes the density values of the density map
    :param density_map: Input density map
    """
    mean = np.mean(density_map.data)
    std = np.std(density_map.data)
    density_map.data = (density_map.data - mean) / std
    return density_map


def get_resolution(map_name: str):
    """
    Get the reported resolution

    :param logger: Logger
    :param map_name: Electron Density Map full name (Example: EMD-0001)
    :return: Reported resolution
    """
    url = 'https://www.ebi.ac.uk/emdb/api/entry/processing/%s' % map_name

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print('Request failed for %s (Status code: %d)' % (url, response.status_code))
        else:
            data = response.json()
            resolution = float(data['processing']['final_reconstruction']['final_reconstruction_type']['resolution']['valueOf_x'])
            return resolution
    except (requests.HTTPError, KeyError, ValueError) as e:
        print('Request failed for %s (Error: %s)' % (url, str(e)))
        return 11


def get_contour_level(map_name: str):
    """
    Get the reported recommended contour level

    :param logger: Logger
    :param map_name: Electron Density Map full name (Example: EMD-0001)
    :return: recommended contour level for map
    """
    map_id = map_name[4:]
    url = "https://www.emdataresource.org/node/solr/emd/select?json.wrf=jQuery331006903074288902045_16" \
          "20453739662&q=id%3A" + Path(map_id).stem + "&_=1620453739663"

    try:
        html = requests.get(url).text
        start = html.find("(") + 1
        end = html.rfind(")")
        jsonData = html[start:end]
        parsed = json.loads(jsonData)

        return parsed["response"]["docs"][0]["mapcontourlevel"]
    except (requests.HTTPError, KeyError, ValueError) as e:
        print('Request failed for %s (Error: %s)' % (url, str(e)))
        return 11

def get_q_score(map_name: str):
    """
    Get the reported recommended contour level

    :param logger: Logger
    :param map_name: Electron Density Map full name (Example: EMD-0001)
    :return: recommended contour level for map
    """
    map_id = map_name[4:]

    url = f'https://www.ebi.ac.uk/emdb/api/analysis/{map_id}?information=all'
    for i in range(3):
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print('Request failed for %s (Status code: %d)' % (url, response.status_code))
            else:
                data = response.json()
                qscore = float(data[str(map_id)]['qscore']['allmodels_average_qscore'])
                print(map_name, qscore)
                return qscore
        except (requests.HTTPError, KeyError, ValueError) as e:
            if i == 2:
                print('Request failed for %s (Error: %s)' % (url, str(e)))
                return -1


def create_hdf5(dataset_dir, size=64):
    path = os.path.join(dataset_dir, 'SR_CryoEM_Autoencoder_Dataset.hdf5')
    with h5py.File(path, 'w') as f:
        for group in ['train', 'val', 'test']:
            group = f.create_group(group)
            group.create_dataset('cubes', (0, 1, size, size, size), maxshape=(None, 1, size, size, size), dtype=np.float32)
    print(f'Created hdf5 file at {path}.')


def add_simulated_data(pdb_path, hdf5, split=None):
    if split is None:
        split = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    cube_list = list()
    pdb = pdb_path.split('/')[-2]
    for res in range(1, 10):
        print(f'Generating simulated data for {pdb} at resolution {res}.')
        density_map = simulate_new(pdb_path, res, 1)
        density_map = normalize_map(density_map)
        cubes = split_map(density_map.data)
        cube_list += cubes
    n = len(cube_list)
    shuffle = np.random.permutation(n)
    cube_list = np.array(cube_list)[shuffle]
    prev = 0.0
    for group, ratio in split.items():
        g = hdf5[group]
        cur = prev + ratio
        start = int(prev * n)
        end = int(cur * n)
        size = end - start
        ori = g['cubes'].shape[0]
        g['cubes'].resize(ori + size, axis=0)
        g['cubes'][-size:] = cube_list[start:end]
        prev = cur


def add_real_data(map_path, pdb_path, resolution, hdf5, split=None):
    emd = map_path.split('/')[-2]
    print(f'Processing {emd} with resolution {resolution}...')
    if split is None:
        split = {'train': 1.0, 'val': 0.0, 'test': 0.0}
    cube_list = list()
    in_map = DensityMap.open(map_path)
    contour = get_contour_level(emd)
    print(f'Get contour level: {contour}')
    in_map = resample(in_map, voxel_size=1.0, contour=contour)
    in_map = normalize_map(in_map)
    in_data = split_map(in_map.data)
    cube_list += in_data
    n = len(cube_list)
    shuffle = np.random.permutation(n)
    cube_list = np.array(cube_list)[shuffle]
    prev = 0.0
    for group, ratio in split.items():
        if ratio == 0.0:
            continue
        g = hdf5[group]
        cur = prev + ratio
        start = int(prev * n)
        end = int(cur * n)
        size = end - start
        ori = g['cubes'].shape[0]
        g['cubes'].resize(ori + size, axis=0)
        g['cubes'][-size:] = cube_list[start:end]
        prev = cur
    print(f'Added {emd} dataset to hdf5')


def construct_dataset(dataset_dir: str, output_dir=None):
    if output_dir is None:
        output_dir = dataset_dir

    train_stats = os.path.join(output_dir, 'Training_dataset_stats_qscore_trimmed.csv')
    if os.path.isfile(train_stats):
        resolutions = dict()
        for i, row in pd.read_csv(train_stats).iterrows():
            resolutions[row['EMD']] = float(row['Resolution'])
    else:
        resolutions = None
    hdf5 = h5py.File(os.path.join(dataset_dir, 'SR_CryoEM_Autoencoder_Dataset.hdf5'), 'a')

    it = 0
    for emd_name in os.listdir(dataset_dir):
        map_dir = os.path.join(dataset_dir, emd_name)
        if not os.path.isdir(map_dir):
            continue
        it += 1
        input_files = os.listdir(map_dir)
        mrc_files = [f for f in input_files if f[:3] in ['emd']]
        pdb_files = [f for f in input_files if f[-3:] in ['pdb', 'ent']]
        r = get_resolution(emd_name) if resolutions == None else resolutions[emd_name]
        add_real_data(os.path.join(map_dir, mrc_files[0]), os.path.join(map_dir, pdb_files[0]), r, hdf5)
        add_simulated_data(os.path.join(map_dir, pdb_files[0]), hdf5)
    hdf5.close()
    print('Dataset construction finished.')
    print('Added %s density map.' % it)
    print('Dataset size: %s' % os.path.getsize(os.path.join(dataset_dir, 'SR_CryoEM_Dataset.hdf5')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/mattovan/SR-CryoEM/SR_CryoEM_Dataset')
    parser.add_argument('--redo_preprocessing', action='store_true', default=True)
    parser.add_argument('--cubic_size', type=int, default=64)
    args = parser.parse_args()
    create_hdf5(args.dataset_path, args.cubic_size)
    construct_dataset(args.dataset_path)
