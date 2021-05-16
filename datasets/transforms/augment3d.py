#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
import scipy

def rotx(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def pc2obj(pc, filepath='test.obj'):
    pc = pc.T
    nverts = pc.shape[1]
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in range(nverts):
            f.write("v %.4f %.4f %.4f\n" % (pc[0,v],pc[1,v],pc[2,v]))

def write_ply_color(points, colors, out_filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    N = points.shape[0]
    save_dir="viz/"
    fout = open(save_dir+out_filename, 'w')
    ### Write header here
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex %d\n" % N)
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    fout.write("end_header\n")
    for i in range(N):
        #c = pyplot.cm.hsv(labels[i])
        c = colors[i,:]
        c = [int(x*150) for x in c]
        fout.write('%f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[0],c[0]))
    fout.close()
            
def write_ply_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i,:]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()
            
def check_aspect(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2])/np.max(crop_range[:2])
    xz_aspect = np.min(crop_range[[0,2]])/np.max(crop_range[[0,2]])
    yz_aspect = np.min(crop_range[1:])/np.max(crop_range[1:])
    return (xy_aspect >= aspect_min) or (xz_aspect >= aspect_min) or (yz_aspect >= aspect_min)

def check_aspect2D(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2])/np.max(crop_range[:2])
    return (xy_aspect >= aspect_min)

def elastic_distortion(coords, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

      pointcloud: numpy array of (number of points, at least 3 spatial dims)
      granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
      magnitude: noise multiplier
    """
    granularity = float(granularity)
    magnitude = float(magnitude)

    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
      noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                   (noise_dim - 2), noise_dim)
    ]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    coords += interp(coords) * magnitude
    return coords

def get_transform3d(data, input_transforms_list, representation="", vox=False):
    output_transforms = []
    ptdata = data['data']
    outdata = []
    counter = 0
    centers = []
    for i, point_cloud in enumerate(ptdata):
        idx=data['label'][i] # Point cloud index, to identify and debug
        if len(point_cloud) > 50000:
            newidx = np.random.choice(len(point_cloud), 50000, replace=False)
            point_cloud = point_cloud[newidx,:]

        #write_ply_color(point_cloud[:,:3], point_cloud[:,3:],str(idx)+"_before.ply")

        for transform_config in input_transforms_list:
            if transform_config['name'] == 'subcenter':
                xyz_center = np.expand_dims(np.mean(point_cloud[:,:3], axis=0), 0)
                point_cloud[:,:3] = point_cloud[:,:3] - xyz_center
            if transform_config['name'] == 'RandomFlipLidar':
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
            if transform_config['name'] == 'RandomRotateLidar':
                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random()*np.pi/2) - np.pi/4 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            if transform_config['name'] == 'RandomScaleLidar':
                point_cloud[:,0:3] = point_cloud[:,0:3] * np.random.uniform(0.95, 1.05)
            if (transform_config['name'] == 'elasticdistortion') and (vox == False):
                #write_ply_color(point_cloud[:,:3], point_cloud[:,3:],str(idx)+"_"+representation+"_before.ply")
                for gran,magn in zip(transform_config['granularity'],transform_config['magnitude']):
                    point_cloud[:,0:3] = elastic_distortion(point_cloud[:,0:3],gran,magn)
                #write_ply_color(point_cloud[:,:3], point_cloud[:,3:],str(idx)+"_"+representation+"_after.ply")
                # for granularity in [2,1,0.5,0.25]:
                #     for magnitude in [0.25,0.5,1,2]:
                #         point_cloud2 = np.copy(point_cloud)
                #         point_cloud2[:,0:3] = elastic_distortion(point_cloud2[:,0:3],granularity,magnitude)
                #         write_ply_color(point_cloud2[:,:3], point_cloud2[:,3:],str(idx)+"_"+representation+"_after_"+str(granularity)+"_"+str(magnitude)+".ply")
            if transform_config['name'] == 'randomcuboidLidar':
                range_xyz = np.max(point_cloud[:,0:2], axis=0) - np.min(point_cloud[:,0:2], axis=0)
                if ('randcrop' in transform_config):
                    crop_range = float(transform_config['crop']) + np.random.rand(2) * (float(transform_config['randcrop']) - float(transform_config['crop']))
                    if ('aspect' in transform_config):
                        loop_count = 0
                        while not check_aspect2D(crop_range, float(transform_config['aspect'])):
                            loop_count += 1
                            crop_range = float(transform_config['crop']) + np.random.rand(2) * (float(transform_config['randcrop']) - float(transform_config['crop']))
                            if loop_count > 100:
                                break
                else:
                    crop_range = float(transform_config['crop'])

                loop_count = 0
                while True:
                    loop_count += 1
                    
                    sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]

                    new_range = range_xyz * crop_range / 2.0

                    max_xyz = sample_center[0:2] + new_range
                    min_xyz = sample_center[0:2] - new_range

                    upper_idx = np.sum((point_cloud[:,0:2] <= max_xyz).astype(np.int32), 1) == 2
                    lower_idx = np.sum((point_cloud[:,0:2] >= min_xyz).astype(np.int32), 1) == 2

                    new_pointidx = (upper_idx) & (lower_idx)
                
                    if (loop_count > 100) or (np.sum(new_pointidx) > float(transform_config['npoints'])):
                        break
                
                point_cloud = point_cloud[new_pointidx,:]
            if transform_config['name'] == 'ToTensorLidar':

                #write_ply_color(point_cloud[:,:3], point_cloud[:,3:],str(idx)+"_after.ply")

                lpt = len(point_cloud)
                if (vox == False):
                    num_points = 16384
                else:
                    num_points = lpt

                points = point_cloud
                if num_points < len(points):
                    pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
                    pts_near_flag = pts_depth < 40.0
                    far_idxs_choice = np.where(pts_near_flag == 0)[0]
                    near_idxs = np.where(pts_near_flag == 1)[0]
                    near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                    choice = []
                    if num_points > len(far_idxs_choice):
                        near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) if len(far_idxs_choice) > 0 else near_idxs_choice
                    else: 
                        choice = np.arange(0, len(points), dtype=np.int32)
                        choice = np.random.choice(choice, num_points, replace=False)
                    np.random.shuffle(choice)
                else:
                    choice = np.arange(0, len(points), dtype=np.int32)
                    if num_points > len(points):
                        if (num_points - len(points)) <= len(choice):
                            extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                        else:
                            extra_choice = np.random.choice(choice, num_points - len(points), replace=True)
                        choice = np.concatenate((choice, extra_choice), axis=0)
                    np.random.shuffle(choice)                    

                point_cloud = point_cloud[choice,:]
                if (vox == False):
                    point_cloud = torch.tensor(point_cloud).float()
            if transform_config['name'] == 'RandomFlip':
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
            if transform_config['name'] == 'RandomRotate':
                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            if transform_config['name'] == 'RandomRotateAll':
                # Rotation along up-axis/Z-axis
                if True:
                    ### Use random rotate for all representation
                    rot_angle = (np.random.random()*np.pi*2) - np.pi # -5 ~ +5 degree
                    if np.random.random() <= 0.33:
                        rot_mat = rotx(rot_angle)
                    elif np.random.random() <= 0.66:
                        rot_mat = roty(rot_angle)
                    else:
                        rot_mat = rotz(rot_angle)
                    point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                else:
                    rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                    rot_mat = rotz(rot_angle)
                    point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            if (transform_config['name'] == 'RandomScale'):
                point_cloud[:,0:3] = point_cloud[:,0:3] * np.random.uniform(0.8, 1.2)
            if transform_config['name'] == 'ColorJitter':
                rgb_color = point_cloud[:,3:6] #+ MEAN_COLOR_RGB
                rgb_color *= (1+0.4*np.random.random(3)-0.2) # brightness change for each channel
                rgb_color += (0.1*np.random.random(3)-0.05) # color shift for each channel
                rgb_color += np.expand_dims((0.05*np.random.random(point_cloud.shape[0])-0.025), -1) # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # 20% gray scale
                random_idx = np.random.choice(rgb_color.shape[0], rgb_color.shape[0]//5, replace=False)
                rgb_color[random_idx] = np.stack([np.dot(rgb_color[random_idx],np.array([0.3,0.59,0.11])), np.dot(rgb_color[random_idx],np.array([0.3,0.59,0.11])), np.dot(rgb_color[random_idx],np.array([0.3,0.59,0.11]))], axis=-1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0])>0.3,-1)
                point_cloud[:,3:6] = rgb_color - 0.5 ### Subtract mean color
            if (transform_config['name'] == 'RandomNoise') and (vox == False):
                pt_shape = point_cloud.shape
                point_noise = (np.random.rand(pt_shape[0], 3) - 0.5) * float(transform_config['noise'])
                point_cloud[:,0:3] += point_noise#[new_pointidx,:]
            
            if (transform_config['name'] == 'RandomNoiseLidar') and (vox == False):
                pt_shape = point_cloud.shape

                point_posnoise = (np.random.rand(pt_shape[0], 3) - 0.5) * 2 * float(transform_config['posnoise'])
                point_cloud[:,0:3] += point_posnoise#[new_pointidx,:]

                point_lightnoise = (np.random.rand(pt_shape[0]) - 0.5) * 2 * float(transform_config['lightnoise'])
                point_cloud[:,3] += point_lightnoise#[new_pointidx,:]
                point_cloud[:,3] = np.clip(point_cloud[:,3], 0, 1)

                #print(str(idx)+" "+representation+": "+ str(point_lightnoise[0]))
            
            if transform_config['name'] == 'randomcuboid':
                range_xyz = np.max(point_cloud[:,0:3], axis=0) - np.min(point_cloud[:,0:3], axis=0)
                if ('randcrop' in transform_config):# and (int(transform_config['randcrop']) == 1):
                    crop_range = float(transform_config['crop']) + np.random.rand(3) * (float(transform_config['randcrop']) - float(transform_config['crop']))
                    if ('aspect' in transform_config):
                        loop_count = 0
                        while not check_aspect(crop_range, float(transform_config['aspect'])):
                            loop_count += 1
                            crop_range = float(transform_config['crop']) + np.random.rand(3) * (float(transform_config['randcrop']) - float(transform_config['crop']))
                            if loop_count > 100:
                                break
                else:
                    crop_range = float(transform_config['crop'])

                skip_step = False
                loop_count = 0
        
                ### Optional for depth selection croption
                if "dist_sample" in transform_config:
                    numb,numv = np.histogram(point_cloud[:,2])
                    max_idx = np.argmax(numb)
                    minidx = max(0,max_idx-2)
                    maxidx = min(len(numv)-1,max_idx+2)
                    range_v = [numv[minidx], numv[maxidx]]
                while True:
                    loop_count += 1
                 
                    sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]
                    if "dist_sample" in transform_config:
                        if (loop_count <= 100):
                            if (sample_center[-1] <= range_v[1]) and (sample_center[-1] >= range_v[0]):
                                continue
                    
                    new_range = range_xyz * crop_range / 2.0

                    max_xyz = sample_center + new_range
                    min_xyz = sample_center - new_range

                    upper_idx = np.sum((point_cloud[:,0:3] <= max_xyz).astype(np.int32), 1) == 3
                    lower_idx = np.sum((point_cloud[:,0:3] >= min_xyz).astype(np.int32), 1) == 3

                    new_pointidx = (upper_idx) & (lower_idx)
                    
                    if (loop_count > 100) or (np.sum(new_pointidx) > float(transform_config['npoints'])):
                        break
                    
                #print ("final", np.sum(new_pointidx))
                point_cloud = point_cloud[new_pointidx,:]
            if transform_config['name'] == 'multiscale':
                rand_scale = [5000, 10000, 15000, 20000]
                rand_scale_idx = np.random.choice(len(rand_scale))
                if len(point_cloud) >= rand_scale[rand_scale_idx]:
                    idx = np.random.choice(len(point_cloud), rand_scale[rand_scale_idx], replace=False)
                else:
                    idx = np.random.choice(len(point_cloud), rand_scale[rand_scale_idx], replace=True)
                point_cloud = point_cloud[idx,:]
                #pc2obj(point_cloud, "new.obj")
            if transform_config['name'] == 'randomdrop':
                range_xyz = np.max(point_cloud[:,0:3], axis=0) - np.min(point_cloud[:,0:3], axis=0)
                drop_base = float(transform_config['dropbase'])
                drop_limit = float(transform_config['droplimit'])
                
                #write_ply_color(point_cloud[:,:3], point_cloud[:,3:],str(idx)+"_before.ply")

                for i in range(int(transform_config['patches'])):
                    new_range = range_xyz * (drop_base + (np.random.random()*2.0-1.0)*drop_limit) / 2.0
                    
                    if np.random.random() <= 0.33:
                        aux=new_range[0]
                        new_range[0]=new_range[1]
                        new_range[1]=aux
                        new_range*=0.8
                    
                    sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]
                    max_xyz = sample_center + new_range
                    min_xyz = sample_center - new_range

                    upper_idx = np.sum((point_cloud[:,0:3] < max_xyz).astype(np.int32), 1) == 3
                    lower_idx = np.sum((point_cloud[:,0:3] > min_xyz).astype(np.int32), 1) == 3

                    new_pointidx = ~((upper_idx) & (lower_idx))
                    point_cloud = point_cloud[new_pointidx,:]

                #write_ply_color(point_cloud[:,:3], point_cloud[:,3:],str(idx)+"_"+representation+".ply")
            
            if transform_config['name'] == 'ToTensor':
                if len(point_cloud) >= 20000:
                    idx = np.random.choice(len(point_cloud), 20000, replace=False)
                else:
                    idx = np.random.choice(len(point_cloud), 20000, replace=True)

                if np.sum(point_cloud) == 0: ### If there are no points, use sudo points
                    pt_shape = point_cloud.shape
                    point_noise = (np.random.rand(pt_shape[0], 3) - 0.5) * float(transform_config['noise'])
                    point_cloud[:,0:3] += point_noise
                point_cloud = point_cloud[idx,:]
                if (vox == False):
                    point_cloud = torch.tensor(point_cloud).float()
            if transform_config['name'] == 'ToFinal':
                if len(point_cloud) >= 20000:
                    idx = np.random.choice(len(point_cloud), 20000, replace=False)
                else:
                    idx = np.random.choice(len(point_cloud), 20000, replace=True)
                    
                if np.sum(point_cloud) == 0:### If there are no points, use sudo points
                    pt_shape = point_cloud.shape
                    point_noise = (np.random.rand(pt_shape[0], 3) - 0.5) * float(transform_config['noise'])
                    point_cloud[:,0:3] += point_noise
                point_cloud = point_cloud[idx,:]
        
        #write_ply_color(point_cloud[:,:3], point_cloud[:,3:],str(idx)+"_"+representation+".ply")
        
        outdata.append(point_cloud)
    
    data['data'] = outdata
    return data
