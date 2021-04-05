# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import os
import numpy as np
import glob 
import pathlib

kitti_dir = '/ctm-hdd-pool01/baraujo/kitti/dc'
frames = glob.glob(kitti_dir+'/training/velodyne'+'/*')

datalist = []

for frame in frames:
    datalist.append(os.path.abspath(frame))

np.save(kitti_dir+'/kitti.npy', datalist)
