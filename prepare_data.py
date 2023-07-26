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




# class CryoEMDataset(Dataset):
#     def __init__(self, in_map, out_map, in_label, out_label):
#         self.in_map = in_map
#         self.out_map = out_map
#         self.in_label = in_label
#         self.out_label = out_label
#
#     def __len__(self):
#         assert len(self.in_map) == len(self.in_label) == len(self.out_map) == len(self.out_label)
#         return len(self.in_map)
#
#     def __getitem__(self, idx):
#         in_map = self.in_map[idx]
#         out_map = self.out_map[idx]
#         in_label = self.in_label[idx]
#         out_label = self.out_label[idx]
#         return in_map, in_label, out_map, out_label


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


class CryoEMDataset(Dataset):
    def __init__(self, hdf5):
        self.in_map = hdf5['in_map']
        self.out_map = hdf5['out_map']
        self.in_label = hdf5['in_label']
        self.out_label = hdf5['out_label']

    def __len__(self):
        assert len(self.in_map) == len(self.in_label) == len(self.out_map) == len(self.out_label)
        return len(self.in_map)

    def __getitem__(self, idx):
        in_map = torch.tensor(self.in_map[idx])
        out_map = torch.tensor(self.out_map[idx])
        in_label = torch.tensor(self.in_label[idx])
        out_label = torch.tensor(self.out_label[idx])
        return in_map, in_label, out_map, out_label


def normalize_map(density_map: DensityMap) -> DensityMap:
    """
    Normalizes the density values of the density map
    :param density_map: Input density map
    """
    density_map.data[density_map.data < 0] = 0
    percentile = np.percentile(density_map.data[np.nonzero(density_map.data)], 99.9)
    density_map.data /= percentile
    density_map.data[density_map.data > 1] = 1

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
    path = os.path.join(dataset_dir, 'SR_CryoEM_Dataset.hdf5')
    with h5py.File(path, 'w') as f:
        for group in ['train', 'val', 'test']:
            group = f.create_group(group)
            group.create_dataset('in_map', (0, 1, size, size, size), maxshape=(None, 1, size, size, size), dtype=np.float32)
            group.create_dataset('out_map', (0, 1, size, size, size), maxshape=(None, 1, size, size, size), dtype=np.float32)
            group.create_dataset('in_label', (0,), maxshape=(None,),dtype=np.float32)
            group.create_dataset('out_label', (0,), maxshape=(None,), dtype=np.float32)
    print(f'Created hdf5 file at {path}.')


def add_simulated_data(pdb_path, hdf5, split=None):
    if split is None:
        split = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    in_map_list = list()
    out_map_list = list()
    in_label = list()
    out_label = list()
    for out_res in range(1, 4):
        out_map = simulate_new(pdb_path, out_res, 1)
        out_map = normalize_map(out_map)
        out_data = split_map(out_map.data)
        tmp_dir = tempfile.mkdtemp(prefix='deeptracer_preprocessing')
        tmp_map_path = os.path.join(tmp_dir, 'out.mrc')
        out_map.save(tmp_map_path)
        for in_res in range(out_res, 10):
            in_map = simulate_on_grid(pdb_path, in_res + 1 + random.random(), tmp_map_path)
            in_map = normalize_map(in_map)
            in_data = split_map(in_map.data)
            in_map_list += in_data
            out_map_list += out_data
            in_label.append(np.full((len(in_data),), in_res, dtype=np.float32))
            out_label.append(np.full((len(out_data),), out_res, dtype=np.float32))
        shutil.rmtree(tmp_dir)
    n = len(in_map_list)
    shuffle = np.random.permutation(n)
    in_map_list = np.array(in_map_list)[shuffle]
    out_map_list = np.array(out_map_list)[shuffle]
    in_label = np.concatenate(in_label, 0)[shuffle]
    out_label = np.concatenate(out_label, 0)[shuffle]
    prev = 0.0
    for group, ratio in split.items():
        cur = prev + ratio
        start = int(prev * n)
        end = int(cur * n)
        size = end - start
        ori = hdf5[group]['in_map'].shape[0]
        hdf5[group]['in_map'].resize(ori + size, axis=0)
        hdf5[group]['in_map'][-size:] = in_map_list[start:end]
        hdf5[group]['out_map'].resize(ori + size, axis=0)
        hdf5[group]['out_map'][-size:] = out_map_list[start:end]
        hdf5[group]['in_label'].resize(ori + size, axis=0)
        hdf5[group]['in_label'][-size:] = in_label[start:end]
        hdf5[group]['out_label'].resize(ori + size, axis=0)
        hdf5[group]['out_label'][-size:] = out_label[start:end]
        prev = cur


def add_real_data(map_path, pdb_path, resolution, hdf5, split=None):
    emd = map_path.split('/')[-2]
    print(f'Processing {emd} with resolution {resolution}...')
    if split is None:
        split = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    in_map_list = list()
    out_map_list = list()
    in_label = list()
    out_label = list()
    in_map = DensityMap.open(map_path)
    contour = get_contour_level(emd)
    print(f'Get contour level: {contour}')
    in_map = resample(in_map, voxel_size=1.0, contour=contour)
    in_map = normalize_map(in_map)
    in_data = split_map(in_map.data)
    tmp_dir = tempfile.mkdtemp(prefix='deeptracer_preprocessing')
    tmp_map_path = os.path.join(tmp_dir, 'in.mrc')
    in_map.save(tmp_map_path)
    for out_res in range(1, int(resolution)):
        out_map = simulate_on_grid(pdb_path, out_res, tmp_map_path)
        out_map = normalize_map(out_map)
        out_data = split_map(out_map.data)
        in_map_list += in_data
        out_map_list += out_data
        in_label.append(np.full((len(in_data),), resolution, dtype= np.float32))
        out_label.append(np.full((len(out_data),), out_res, dtype=np.float32))
        print(f'Simulated resolution {out_res} for {emd}.')
    shutil.rmtree(tmp_dir)

    n = len(in_map_list)
    shuffle = np.random.permutation(n)
    in_map_list = np.array(in_map_list)[shuffle]
    out_map_list = np.array(out_map_list)[shuffle]
    in_label = np.concatenate(in_label, 0)[shuffle]
    out_label = np.concatenate(out_label, 0)[shuffle]
    prev = 0.0
    for group, ratio in split.items():
        g = hdf5[group]
        cur = prev + ratio
        start = int(prev * n)
        end = int(cur * n)
        size = end - start
        ori = g['in_map'].shape[0]
        g['in_map'].resize(ori + size, axis=0)
        g['in_map'][-size:] = in_map_list[start:end]
        g['out_map'].resize(ori + size, axis=0)
        g['out_map'][-size:] = out_map_list[start:end]
        g['in_label'].resize(ori + size, axis=0)
        g['in_label'][-size:] = in_label[start:end]
        g['out_label'].resize(ori + size, axis=0)
        g['out_label'][-size:] = out_label[start:end]
        prev = cur
        assert g['in_map'].shape[0] == g['out_map'].shape[0] == g['in_label'].shape[0] == g['out_label'].shape[0]
    print(f'Added {emd} dataset to hdf5')


def examine_dataset(dataset_dir: str, output_dir=None, redo=False):
    """
    Normalize original map (X) and create a theoretical map (Y) from predicted structure.

    :param dataset_dir: Directory of un-processed dataset
    :param output_dir: output_dir
    :param redo: re-prepare
    """
    if output_dir is None:
        output_dir = dataset_dir

    train_stats = os.path.join(output_dir, 'Training_dataset_stats.csv')
    if not os.path.exists(train_stats) or redo:
        with open(train_stats, 'w+') as file:
            csvWriter = csv.writer(file)
            csvWriter.writerow(['EMD', 'Resolution', 'similarity', 'file_size'])

    exist = set(pd.read_csv(train_stats, index_col=0).index)
    args = list()
    for emd_name in os.listdir(dataset_dir):
        map_dir = os.path.join(dataset_dir, emd_name)
        if not os.path.isdir(map_dir) or emd_name in exist:
            continue

        input_files = os.listdir(map_dir)
        if redo:
            mrc_files = [f for f in input_files if f[:3] in ['emd']]
        else:
            mrc_files = [f for f in input_files if f[-3:] in ['mrc', 'map']]
        pdb_files = [f for f in input_files if f[-3:] in ['pdb', 'ent']]

        if redo or len(mrc_files) == 1:
            args.append([os.path.join(map_dir, mrc_files[0]),
                         os.path.join(map_dir, pdb_files[0]), os.path.join(output_dir, emd_name), train_stats])

    print('Pre-processing %d density maps' % len(args))
    with Pool(cpu_count()) as pool:
        pool.starmap(similarity, args)


def similarity(density_map_path: str, solved_structure_path: str, output_dir: str, csv_file=None):
    """
    Calculate similarity between density map and solved structure and save the results into csv file.

    :param density_map_path: Path of density map
    :param solved_structure_path: Path of solved structure
    :param output_dir: Directory where map output are saved
    :param csv_file: CSV file to save statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    emd_name = output_dir.split('/')[-1]

    X = DensityMap.open(density_map_path)
    size = X.voxel_size.x * X.voxel_size.y * X.voxel_size.z * os.path.getsize(density_map_path)
    if size > 200 * 1024 * 1024:
        print(f'Map size too large: {emd_name}, size {size}')
        return
    resolution = get_resolution(emd_name)
    contour_level = get_contour_level(emd_name)
    map_max = np.percentile(X.data[np.nonzero(X.data)], 99.99)
    x_percentile = X.data[X.data > contour_level].size / X.data.size * 100.0

    # Normalize and re-sample on grid size 1A
    X.data[X.data < 0] = 0
    X.data /= map_max
    X.data[X.data > 1] = 1
    X = resample(X, voxel_size=1.0)

    tmp_dir = tempfile.mkdtemp(prefix='deeptracer_preprocessing')
    tmp_map_path = os.path.join(tmp_dir, 'temp.mrc')
    X.save(tmp_map_path)
    Chimera.run(tmp_dir, [
        'open %s' % solved_structure_path,
        'open %s' % tmp_map_path,
        'volume #1 step 1',
        'molmap #0 %s onGrid #1' % resolution,
        'volume #2 save %s' % tmp_map_path,
    ])
    Y = DensityMap.open(tmp_map_path)
    shutil.rmtree(tmp_dir)

    Y.data[Y.data < 0.0039] = 0
    y_percentile = Y.data[Y.data > 0].size / Y.data.size * 100.0
    percentile = max(x_percentile, y_percentile)
    threshold = np.percentile(X.data, 100.0 - percentile * 2)
    print("Correlation examination threshold is %s" % threshold)

    Y.data /= np.percentile(Y.data[np.nonzero(Y.data)], 99.99)
    Y.data[Y.data > 1] = 1
    correlation = np.corrcoef(X.data[X.data > threshold], Y.data[X.data > threshold])
    print("Correlation coefficient is %s" % correlation[0, 1])

    with open(csv_file, 'a') as file:
        csvWriter = csv.writer(file)
        csvWriter.writerow([emd_name, resolution, correlation[0, 1], size // 1024 // 1024])


def process_q_score(dataset_path):
    path = os.path.join(dataset_path, 'Training_dataset_stats.csv')
    def worker(row):
        if row['qscore'] == -1:
            row['qscore'] = get_q_score(row['EMD'])
        return row

    df = pd.read_csv(path)
    with Pool(cpu_count()) as p:
        result = p.map(worker, [row for _, row in df.iterrows()])

    df = pd.DataFrame(result)
    df.to_csv(os.path.join(args.dataset_path, 'Training_dataset_stats_qscore.csv'), index=False)


def trim_data(dataset_path):
    path = os.path.join(dataset_path, 'Training_dataset_stats_qscore.csv')
    df = pd.read_csv(path)
    df = df[df['similarity'] >= 0.7]
    def f(x):
        return 1 / 200 * (x - 11) ** 2 + 0.04
    def g(x):
        return 0.42 * np.exp(-(1 / 6) * x)
    df = df[df['qscore'] >= g(df['Resolution'])]
    df.to_csv(os.path.join(args.dataset_path, 'Training_dataset_stats_qscore_trimmed.csv'), index=False)


def remove_trimmed_data(dataset_path):
    path = os.path.join(dataset_path, 'Training_dataset_stats_qscore_trimmed.csv')
    df = pd.read_csv(path)
    emd_list = set(df['EMD'])
    for emd_name in os.listdir(dataset_path):
        emd_path = os.path.join(dataset_path, emd_name)
        if not os.path.isdir(emd_path):
            continue
        if emd_name not in emd_list:
            shutil.rmtree(emd_path)
            print(f'Removed {emd_name}')


def construct_dataset(dataset_dir: str, output_dir=None):
    if output_dir is None:
        output_dir = dataset_dir

    train_stats = os.path.join(output_dir, 'Training_dataset_stats_qscore_trimmed.csv')
    resolutions = dict()
    for i, row in pd.read_csv(train_stats).iterrows():
        resolutions[row['EMD']] = float(row['Resolution'])
    hdf5 = h5py.File(os.path.join(dataset_dir, 'SR_CryoEM_Dataset.hdf5'), 'a')

    it = 0
    for emd_name in os.listdir(dataset_dir):
        map_dir = os.path.join(dataset_dir, emd_name)
        if not os.path.isdir(map_dir):
            continue
        it += 1
        input_files = os.listdir(map_dir)
        mrc_files = [f for f in input_files if f[:3] in ['emd']]
        pdb_files = [f for f in input_files if f[-3:] in ['pdb', 'ent']]
        add_real_data(os.path.join(map_dir, mrc_files[0]), os.path.join(map_dir, pdb_files[0]), resolutions[emd_name], hdf5)
    hdf5.close()
    print('Dataset construction finished.')
    print('Added %s density map.' % it)
    print('Dataset size: %s' % os.path.getsize(os.path.join(dataset_dir, 'SR_CryoEM_Dataset.hdf5')))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/sbcaesar/SR_CryoEM_Dataset')
    parser.add_argument('--redo_preprocessing', action='store_true', default=True)
    parser.add_argument('--cubic_size', type=int, default=64)
    args = parser.parse_args()

    # examine_dataset(args.dataset_path, redo=args.redo_preprocessing)
    # trim_data(args.dataset_path)
    # remove_trimmed_data(args.dataset_path)
    # create_hdf5(args.dataset_path, args.cubic_size)
    # construct_dataset(args.dataset_path)

