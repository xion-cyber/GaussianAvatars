# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 3/13/25 2:41 PM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : my_train.py
# @desc     :

from gaussian_renderer import render
from plyfile import PlyData, PlyElement
from PIL import Image
import numpy as np
import torch
import sys
import os
from pathlib import Path
import pickle
from typing import NamedTuple, Optional
from utils.graphics_utils import ClothMesh, SmplxMesh, ClothedSmplxMesh, read_ply_mesh
from scene.cloth_smplx_model import ClothSmplxGaussianModel



if __name__ == "__main__":
    render()
