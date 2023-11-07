# Copyright (c) CryoRes Team, ZhangLab, Tsinghua University. All Rights Reserved

import mrcfile
import numpy as np
import torch
import os
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import shutil
from torch.cuda.amp import autocast as autocast
from cryores.utils import vis_mask, vis_localres  
from cryores.model import ResidualUNet3D as ResidualUNet3D_ori
from cryores.model_rescale import ResidualUNet3D

def cut_density(mappath):
    print(mappath)
    mrc = mrcfile.open(mappath)
    data_x_tag = mrc.data
    # print("shape ori", data_x_tag.shape)
    m = data_x_tag.mean()
    u = data_x_tag.std()
    data_x_tag.flags.writeable = True
    data_x_tag[data_x_tag <= m + 3 * u] = -10
    coords = np.array(np.where(data_x_tag > -10))

    shape_x = data_x_tag.shape
    xdend = np.max(coords[0])+1
    ydend = np.max(coords[1])+1
    zdend = np.max(coords[2])+1
    xdenx = np.min(coords[0])
    ydenx = np.min(coords[1])
    zdenx = np.min(coords[2])
    a0 = 1+np.max(coords[0]) - np.min(coords[0])
    a1 = 1+np.max(coords[1]) - np.min(coords[1])
    a2 = 1+np.max(coords[2]) - np.min(coords[2])
    size_x = a0
    size_y = a1
    size_z = a2 
    cover = 1
    if xdend - xdenx >= 250:
        size_x = 248
        cover = 0
    if ydend - ydenx >= 250:
        size_y = 248
        cover = 0
    if zdend - zdenx >= 250:
        size_z = 248
        cover = 0
        
    mrc.close()

    return xdenx, xdend, ydenx, ydend, zdenx, zdend, size_x, size_y, size_z, cover

@torch.no_grad()
def test(model_ori, model_reshape, mappath, device_ori, device_reshape):

    model_ori.eval()
    model_reshape.eval()
    dirname   = os.path.dirname(mappath)
    basename  = os.path.basename(mappath).split('.')[0]
    suffix    = mappath.split(".")[-1]
    localres_path = f"{dirname}/{basename}_localres.{suffix}"
    mask_path = f"{dirname}/{basename}_mask.{suffix}"

    xdenx, xdend, ydenx, ydend, zdenx, zdend, size_x, size_y, size_z, cover = cut_density(mappath)

    mrc = mrcfile.open(mappath)
    data_x = mrc.data
    level = data_x.mean() + 6 * data_x.std()
    shape_ori = data_x.shape
    mrc.close()
    data_x = data_x[xdenx:xdend, ydenx:ydend, zdenx:zdend]
    shape_old = data_x.shape
    # print("cut data_x     ", data_x.shape)
    m = data_x.mean()
    u = data_x.std()
    if u != 0:
        data_x = (data_x - m) / u
    data_x = torch.unsqueeze(torch.tensor(data_x), 0)
    data_x = torch.unsqueeze(data_x, 0)
    
    with autocast():
        a = torch.tensor([
            [
                [[[-5]]],[[[-4.5]]],[[[-4]]],[[[-3.5]]],[[[-3]]],[[[-2.5]]],[[[-2]]],[[[-1.5]]],[[[-1]]],[[[-0.9]]],
                [[[-0.8]]],[[[-0.7]]],[[[-0.6]]],[[[-0.5]]],[[[-0.4]]],[[[-0.3]]],[[[-0.2]]],[[[-0.1]]],[[[0]]],
                [[[0.1]]],[[[0.2]]],[[[0.3]]],[[[0.4]]],[[[0.5]]],[[[0.6]]],[[[0.7]]],[[[0.8]]],[[[0.9]]],[[[1]]],
                [[[1.5]]],[[[2]]],[[[2.5]]],[[[3]]],[[[3.5]]],[[[4]]],[[[4.5]]],[[[5]]]
            ]
        ])
        if cover == 1:
            # print("No down!!")
            data_x = data_x.to(device_ori)
            a = a.to(device_ori)
            try:
                output_local, output, output_class = model_ori(data_x)
            except ValueError as exception:
                exception_str = str(exception)
                size_ok_x = int(exception_str.split("for an input of torch.Size([")[1].split("]")[0].split(",")[0])*2
                size_ok_y = int(exception_str.split("for an input of torch.Size([")[1].split("]")[0].split(",")[1])*2
                size_ok_z = int(exception_str.split("for an input of torch.Size([")[1].split("]")[0].split(",")[2])*2
                xdenx = xdenx + size_x - size_ok_x
                ydenx = ydenx + size_y - size_ok_y
                zdenx = zdenx + size_z - size_ok_z
                data_x = data_x[:,:,(size_x - size_ok_x):, (size_y - size_ok_y):, (size_z - size_ok_z):]
                output_local, output, output_class = model_ori(data_x)
            # else:
                # print("No down no cut twice")
        else:
            # print("Need down!!")
            data_x = F.interpolate(data_x, size=[size_x, size_y, size_z])
            data_x = data_x.to(device_reshape)
            a = a.to(device_reshape)
            try:
                output_local, output, output_class = model_reshape(data_x)
            except Exception as exception:
                exception_str = str(exception)
                size_ok_x = int(exception_str.split("for an input of torch.Size([")[1].split("]")[0].split(",")[0])*2
                size_ok_y = int(exception_str.split("for an input of torch.Size([")[1].split("]")[0].split(",")[1])*2
                size_ok_z = int(exception_str.split("for an input of torch.Size([")[1].split("]")[0].split(",")[2])*2
                data_x = data_x[:,:,(size_x - size_ok_x):, (size_y - size_ok_y):, (size_z - size_ok_z):]
                output_local, output, output_class = model_reshape(data_x)
            # else:
            #     print("No down no cut twice")    
        x_local = torch.mul(a,output_local)
        output_local = torch.sum(x_local, dim=1)
        output_local = torch.unsqueeze(output_local, 0)
        
        if cover == 0:
            output_class = F.interpolate(output_class.cpu().float(), size=shape_old, mode="trilinear")
            output_local = F.interpolate(output_local.cpu().float(), size=shape_old, mode="trilinear")
        output_class = output_class.cpu()
        output_local = output_local.cpu()
        output = output.cpu()
        
        output_class[output_class >= 0] = 1
        output_class[output_class < 0] = 0
        output_local = output_local + output
        output_local[output_class == 0] = 100

        local_res_ma = np.ma.masked_where(output_local == 100, output_local, copy=True)
        minRes_ai = round(np.ma.min(local_res_ma), 3)
        maxRes_ai = round(np.ma.max(local_res_ma), 3)
        
    shutil.copy(mappath, localres_path)
    mrc_r = mrcfile.open(localres_path, mode='r+')
    data_write_y = (torch.ones((shape_ori[0], shape_ori[1], shape_ori[2]))*100)
    data_write_y[xdenx:xdend, ydenx:ydend, zdenx:zdend] = output_local[0][0].cpu()
    mrc_r.data[:] = data_write_y
    mrc_r.close()

    shutil.copy(mappath, mask_path)
    mrc_r_mask = mrcfile.open(mask_path, mode='r+')
    data_write_mask = (torch.ones((shape_ori[0], shape_ori[1], shape_ori[2]))*0)
    data_write_mask[xdenx:xdend, ydenx:ydend, zdenx:zdend] = output_class[0][0].cpu()
    mrc_r_mask.data[:] = data_write_mask
    mrc_r_mask.close()

    output_dict = {
        "global_res": output,
        "local_res_min": minRes_ai,
        "local_res_max": maxRes_ai,
        "map_level": level,
    }
    return output_dict

def prediction(mappath, device):

    model_ori = ResidualUNet3D_ori(in_channels=1, out_channels=1)
    model_reshape = ResidualUNet3D(in_channels=1, out_channels=1)

    if device == -1:
        device_ori = torch.device("cpu")
        device_reshape = torch.device("cpu")
    else:
        device_ori = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
        device_reshape = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
    print("Using GPU: ", device_ori)
    model_ori = model_ori.to(device_ori)
    model_reshape = model_reshape.to(device_reshape)
    
    dir_ori = './ckpt/params.pth'
    checkpoint_ori = torch.load(dir_ori,map_location=device_ori)
    model_ori.load_state_dict(checkpoint_ori['net'])
    dir_reshape = './ckpt/params_rescale.pth'
    checkpoint_reshape = torch.load(dir_reshape,map_location=device_reshape)
    model_reshape.load_state_dict(checkpoint_reshape['net'])
        
    output_dict = test(model_ori, model_reshape, mappath, device_ori, device_reshape)
    print(f"Global resolution: {output_dict['global_res']}")
    print(f"Local  resolution: {output_dict['local_res_min']}~{output_dict['local_res_max']}")
    # generate local resolution visualization commline.
    vis_localres(mappath, output_dict['local_res_min'], output_dict['local_res_max'], output_dict['map_level'])
    # generate mask visualization commline.
    vis_mask(mappath, output_dict['map_level'])
    print("CryoRes inference finished.")


