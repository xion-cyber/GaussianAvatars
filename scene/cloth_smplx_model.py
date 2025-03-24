# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 3/12/25 2:26 PM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : cloth_smplx_model.py
# @desc     :

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from utils.sh_utils import RGB2SH
from .gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import ClothedSmplxMesh
from simple_knn._C import distCUDA2

class ClothSmplxGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, clothed_mesh: ClothedSmplxMesh):
        super().__init__(sh_degree)
        self.clothed_mesh = clothed_mesh
        if self.binding is None:
            self.binding = torch.arange(len(self.clothed_mesh.faces), dtype=torch.int32).cuda()
            # self.binding_counter = torch.ones(len(self.clothed_mesh.faces), dtype=torch.int32).cuda()
            self.init_position = torch.from_numpy(clothed_mesh.centers).to(torch.double).cuda()


    def create_from_mesh(self, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.from_numpy(self.clothed_mesh.centers).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(self.clothed_mesh.center_colors).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        print("Number of points at initialisation: ", self.get_xyz.shape[0])

        # if self.binding is None:
        #     raise NotImplementedError
        # else:
        #     scales = torch.log(torch.ones((self.get_xyz.shape[0], 3), device="cuda"))
        dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def gaussian_shape_regularization(self):
        scaling = torch.exp(self._scaling)
        return torch.linalg.norm(torch.ones_like(scaling)-scaling, ord=2, dim=1).mean()

    def position_anchoring(self, distance_threshold: float):
        distance = torch.linalg.norm(self.get_xyz - self.init_position, ord=2, dim=1)
        msks = distance < distance_threshold
        modified_distance = torch.where(msks, torch.zeros_like(distance), distance)
        return modified_distance.mean()



