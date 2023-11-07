# Copyright (c) CryoRes Team, ZhangLab, Tsinghua University. All Rights Reserved

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from cryores.modules import Encoder, Decoder, DoubleConv, ExtResNetBlock
from cryores.utils import number_of_features_per_level

class Abstract3DUNet(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=4, num_levels=2, is_segmentation=True, testing=False, **kwargs):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=basic_module,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data 
                # isotropic after 1-2 pooling operations
                # currently pools with a constant kernel: (2, 2, 2)
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=basic_module,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)
        self.gn_global_p = nn.GroupNorm(num_groups=num_groups, num_channels=f_maps[-1])
        self.conv_global_p = nn.Conv3d(f_maps[-1], f_maps[-1]*2, 3, bias=False, padding=1)
        self.pooling_max = nn.MaxPool3d(kernel_size=(2,2,2))
        self.gn_global_2_p = nn.GroupNorm(num_groups=num_groups, num_channels=f_maps[-1]*2)
        self.conv_global_2_p = nn.Conv3d(f_maps[-1]*2, f_maps[-1]*4, 3, bias=False, padding=1)
        self.gn_global_3_p = nn.GroupNorm(num_groups=num_groups, num_channels=f_maps[-1]*4)
        self.conv_global_3_p = nn.Conv3d(f_maps[-1]*4, f_maps[-1]*8, 3, padding=1)
        self.active = nn.ReLU(inplace=True)#nn.Sigmoid()
        self.final_conv_global_p = nn.Conv3d(f_maps[-1]*8, f_maps[-1]*4, 1)
        self.final2_conv_global_p = nn.Conv3d(f_maps[-1]*4, 1, 1)
        
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=basic_module,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        self.final_conv_class_bi_new1 = nn.Conv3d(f_maps[0], f_maps[0]*2, 3, bias=False, padding=1)
        self.final_conv_class_bi_new2 = nn.Conv3d(f_maps[0]*2, 1, 1)
        self.final2_conv_local_new1_new = nn.Conv3d(f_maps[0], f_maps[0]*2, 3, bias=False, padding=1)
        self.final2_conv_local_new2_new = nn.Conv3d(f_maps[0]*2, 37, 1)
        

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        x_global = self.pooling_max(self.active(self.conv_global_p(self.gn_global_p(x))))
        x_global = self.pooling_max(self.active(self.conv_global_2_p(self.gn_global_2_p(x_global))))
        x_global = self.active(self.conv_global_3_p(self.gn_global_3_p(x_global)))
        x_global = self.active(self.final_conv_global_p(x_global))
        x_global = self.final2_conv_global_p(x_global)
        x_global = nn.functional.adaptive_avg_pool3d(x_global, (1, 1, 10))
        x_global = F.softmax(x_global, dim=4)
        x_global = x_global[0][0][0][0][0] * 1 + x_global[0][0][0][0][1] * 2 + x_global[0][0][0][0][2] * 3 + \
                   x_global[0][0][0][0][3] * 4 + x_global[0][0][0][0][4] * 5 + x_global[0][0][0][0][5] * 6 + \
                   x_global[0][0][0][0][6] * 7 + x_global[0][0][0][0][7] * 8 + x_global[0][0][0][0][8] * 9 + \
                   x_global[0][0][0][0][9] * 10 

        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x_local = self.final2_conv_local_new1_new(x)
        x_local = self.final2_conv_local_new2_new(self.active(x_local))
        x_local = F.softmax(x_local, dim=1)
        x_class = self.final_conv_class_bi_new1(x)
        x_class = self.final_conv_class_bi_new2(self.active(x_class))

        return x_local, x_global, x_class


class UNet3D(Abstract3DUNet):

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=2, num_levels=4, is_segmentation=True, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     num_groups=num_groups, num_levels=num_levels, is_segmentation=is_segmentation,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=2, num_levels=2, is_segmentation=True, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock, f_maps=f_maps, layer_order=layer_order,
                                             num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             **kwargs)
