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

kitti_dir = '/home/baraujo/kitti'
set= 'val'  #train, val, trainval or test
tsne=True    #if tsne==True: only a small portion of pointclouds is extracted
particular_tsne=True

if particular_tsne:
    
    if set=='train':
        numberlist=['000214',
                    '000955',
                    '001041',
                    '001229',
                    '001288',
                    '001430',
                    '001447',
                    '001648',
                    '003541',
                    '005870',
                    ]
    elif set=='val':
        numberlist=['001054',
                    '001363',
                    '001694',
                    '001800',
                    '002068',
                    '002290',
                    '002411',
                    '003425',
                    '004709',
                    '005368',
                    ]

    pathlist = []

    for nr in numberlist:
        pathlist.append(os.path.join(kitti_dir,'training', 'velodyne', nr+'.bin'))
    
    pathlist=np.array(pathlist)
    numberlist=np.array(numberlist)
    np.save(kitti_dir+'/kitti_tsne_particular_'+set+'.npy', pathlist)
    np.savetxt(kitti_dir+'/kitti_tsne_particular_'+set+'.txt',pathlist,fmt="%s")
    np.savetxt(kitti_dir+'/kitti_tsne_particular_'+set+'_ids.txt',numberlist,fmt="%s")


else:

    pathlist = []
    numberlist = []

    filepath = os.path.join(kitti_dir, 'ImageSets', set+'.txt')
    file = open(filepath)
    for sample in file:
        pathlist.append(os.path.join(kitti_dir,'training', 'velodyne', sample.rstrip() + '.bin'))
        numberlist.append(sample.rstrip())


    if tsne:
        # Choose a multiple of 10 number of samples
        pathlist_temp=np.array(pathlist)
        numberlist_temp=np.array(numberlist)
        np.random.seed(0)
        pathlist=np.random.choice(pathlist_temp,size=10*10,replace=False)
        np.random.seed(0)
        numberlist=np.random.choice(numberlist_temp,size=10*10,replace=False)

        # Save list of paths to .npy and .txt
        np.save(kitti_dir+'/kitti_tsne_'+set+'.npy', pathlist)
        np.savetxt(kitti_dir+'/kitti_tsne_'+set+'.txt',pathlist,fmt="%s")
        np.savetxt(kitti_dir+'/kitti_tsne_'+set+'_ids.txt',numberlist,fmt="%s")

    else:
        # Save list of paths to .npy and .txt
        np.save(kitti_dir+'/kitti_'+set+'.npy', pathlist)
        np.savetxt(kitti_dir+'/kitti_'+set+'.txt',pathlist,fmt="%s")


