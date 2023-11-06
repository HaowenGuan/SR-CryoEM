import os
import argparse
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from cryores.infer_modif import prediction
from util.density_map import DensityMap
from util.map_splitter import split_map


def localres_prediction(mappath, cuda_device=0):
    output_dict = prediction(mappath, cuda_device, args.ckpt_path)
    return output_dict


class ConditioningDataset(Dataset):
    def __init__(self, hdf5):
        self.cubes = hdf5['cubes']

    def __len__(self):
        return len(self.cubes)

    def __getitem__(self, idx):
        cube = torch.tensor(self.cubes[idx])
        return cube


def create_hdf5(dataset_dir, size=64):
    path = os.path.join(dataset_dir, 'SR_CryoEM_LocalRes_dataset.hdf5')
    with h5py.File(path, 'w') as hdf5:
        for group in ['train', 'val', 'test']:
            group = hdf5.create_group(group)
            group.create_dataset('cubes', (0, 1, size, size, size), maxshape=(None, 1, size, size, size),
                                 dtype=np.float32)
    print(f'Created hdf5 file at {path}.')


def add_data(localres_dict, hdf5, split=None):
    localres_path = localres_dict["localres_path"]
    name = localres_path.split('/')[-2]
    print(f'Processing {name} resolution file with global resolution {localres_dict["global_res"]}...')
    if split is None:
        split = {'train': 1.0, 'val': 0.0, 'test': 0.0}
    cube_list = list()
    in_map = DensityMap.open(localres_path)
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
    print(f'Added {name} in {n} cubes to hdf5 file.')


def construct_dataset(dataset_dir, output_dir):
    hdf5 = h5py.File(os.path.join(dataset_dir, 'SR_CryoEM_LocalRes_dataset.hdf5'), 'a')
    cnt = 0
    for file in tqdm(os.listdir(dataset_dir)):
        cnt += 1
        map_dir = os.path.join(dataset_dir, file)
        if not os.path.isdir(map_dir):
            continue
        input_files = os.listdir(map_dir)
        map_files = [f for f in input_files if f[:3] in ['emd']]
        localres_dict = localres_prediction(os.path.join(map_dir, map_files[0]), cuda_device=args.cuda_device)
        add_data(localres_dict, hdf5)
    hdf5.close()
    print(f'Finished constructing dataset at {output_dir}.')
    print('Added {} maps to dataset.'.format(cnt))
    print('Dataset size: {}.'.format(os.path.getsize(os.path.join(dataset_dir, 'SR_CryoEM_LocalRes_dataset.hdf5'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='cryores/ckpt')
    args = parser.parse_args()
    create_hdf5(args.dataset_path, args.size)
    construct_dataset(args.dataset_path, args.dataset_path)
