# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import random
import time
import warnings
import yaml
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import torch.multiprocessing as mp

import utils.logger
from utils import main_utils

parser = argparse.ArgumentParser(description='PyTorch Self Supervised Training in 3D')

parser.add_argument('cfg', help='model directory')
parser.add_argument('ckp', help='checkpoint directory')
parser.add_argument('--quiet', action='store_true')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:14475', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--ngpus', default=8, type=int,
                    help='number of GPUs to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--set', default=None, type=str,
                    help='something')
parser.add_argument('--run', default=None, type=str,
                    help='something')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    print('Dataset:',cfg["dataset"]["DATASET_NAMES"])
    
    #cfg["dataset"]["BATCHSIZE_PER_REPLICA"]=10
    if args.set == 'train':
        cfg["dataset"]["DATA_PATHS"]=['/home/baraujo/kitti/kitti_tsne_train.npy']
    elif args.set == 'val':
        cfg["dataset"]["DATA_PATHS"]=['/home/baraujo/kitti/kitti_tsne_val.npy']
    cfg["num_workers"]=1
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = args.ngpus
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, cfg)
    
def main_worker(gpu, ngpus, args, cfg):
    args.gpu = gpu
    ngpus_per_node = ngpus
    
    # Setup environment
    args = main_utils.initialize_distributed_backend(args, ngpus_per_node) ### Use other method instead
    #logger, tb_writter, model_dir = main_utils.prep_environment(args, cfg)
    model_dir = '{}/{}'.format(cfg['model']['model_dir'], cfg['model']['name'])
    log_fn = '{}/eval.log'.format(model_dir)
    logger = utils.logger.Logger(quiet=args.quiet, log_fn=log_fn, rank=args.rank)

    # Define model
    model = main_utils.build_model(cfg['model'], logger)
    model, args = main_utils.distribute_model_to_cuda(model, args)

    # Define dataloaders
    train_loader = main_utils.build_dataloaders(cfg['dataset'], cfg['num_workers'], args.multiprocessing_distributed, logger)       

    # Define criterion    
    #train_criterion = main_utils.build_criterion(cfg['loss'], logger=logger)
    #train_criterion = train_criterion.cuda()
    train_criterion=None

    # Define optimizer
    #optimizer, scheduler = main_utils.build_optimizer(
    #    params=list(model.parameters())+list(train_criterion.parameters()),
    #    cfg=cfg['optimizer'],
    #    logger=logger)
    optimizer, scheduler = None, None

    ckp_manager = main_utils.CheckpointManager(model_dir, rank=args.rank, dist=args.multiprocessing_distributed)
    # Optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    # if cfg['resume']:
    #     if ckp_manager.checkpoint_exists(last=True):
    try:
        #start_epoch = ckp_manager.restore(fn=args.ckp, reset_optimizer=cfg['optimizer']['reset'], model=model, optimizer=optimizer, train_criterion=train_criterion)
        start_epoch = ckp_manager.restore(fn=args.ckp, reset_optimizer=cfg['optimizer']['reset'], model=model)
        #scheduler.step(start_epoch)
        logger.add_line("Checkpoint loaded: '{}' (epoch {})".format(args.ckp, start_epoch))
    except:
        logger.add_line("No checkpoint found at '{}'".format(args.ckp))

    logger.add_line("Data path: '{}'".format(cfg["dataset"]["DATA_PATHS"]))
    logger.add_line("Run: '{}'".format(args.run))
    logger.add_line("Set: '{}'".format(args.set))


    cudnn.benchmark = True
    #do_checkpoint = False
    
    X = []
    idx = []

    ############################ TRAIN #########################################
    test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1
    for epoch in range(30):
        #if (epoch % 10) == 0 and do_checkpoint == True:
        #    ckp_manager.save(epoch, model=model, train_criterion=train_criterion, optimizer=optimizer, filename='checkpoint-ep{}.pth.tar'.format(epoch))

        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
        #logger.add_line('LR: {}'.format(scheduler.get_lr()))
        run_phase('train', train_loader, model, optimizer, train_criterion, epoch, args, cfg, logger, X, idx)
        #scheduler.step(epoch)

        # do_checkpoint = True

        #if ((epoch % test_freq) == 0) or (epoch == end_epoch - 1):
        #    ckp_manager.save(epoch+1, model=model, optimizer=optimizer, train_criterion=train_criterion)

    np.save('tsne/data/animation/embeds_'+args.run+'_'+args.set+'.npy', np.array(X).reshape(-1,128))
    np.save('tsne/data/animation/labels_'+args.run+'_'+args.set+'.npy', np.array(idx).ravel())

def run_phase(phase, loader, model, optimizer, criterion, epoch, args, cfg, logger, X, idx):
    # from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    # batch_time = metrics_utils.AverageMeter('Time', ':6.3f', window_size=100)
    # data_time = metrics_utils.AverageMeter('Data', ':6.3f', window_size=100)
    # loss_meter = metrics_utils.AverageMeter('Loss', ':.3e')
    # loss_meter_npid1 = metrics_utils.AverageMeter('Loss_npid1', ':.3e')
    # loss_meter_npid2 = metrics_utils.AverageMeter('Loss_npid2', ':.3e')
    # loss_meter_cmc1 = metrics_utils.AverageMeter('Loss_cmc1', ':.3e')
    # loss_meter_cmc2 = metrics_utils.AverageMeter('Loss_cmc2', ':.3e')
    # progress = utils.logger.ProgressMeter(len(loader), [batch_time, data_time, loss_meter, loss_meter_npid1, loss_meter_npid2, loss_meter_cmc1, loss_meter_cmc2], phase=phase, epoch=epoch, logger=logger)

    # switch to train mode
    #model.train(phase == 'train')

    # end = time.time()
    # device = args.gpu if args.gpu is not None else 0
    for i, sample in enumerate(loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        # if phase == 'train':
        #     embedding = model(sample)
        # else:
        logger.add_line('{}'.format(i))
        label = sample["label"].numpy()
        #print("Loaded ",label)
        
        #inters = np.intersect1d(indices, label)

        #for index in label: 

            #print("-> Loader returned ",index)

        with torch.no_grad():
            embedding = model(sample)

        npembed = np.array([embed.cpu().detach().numpy() for embed in embedding])
        
        # #if np.where(label==index)[0][0]==0:
        # npembed_del = np.delete(npembed,1,axis=1)
        # npembed_del_shaped = npembed_del.reshape((4, 128))
        # X.append(npembed_del_shaped)
        # idx.append(label[0])

        # #else:
        # npembed_del = np.delete(npembed,0,axis=1)
        # npembed_del_shaped = npembed_del.reshape((4, 128)) # Even indices are representation of first sample
        # X.append(npembed_del_shaped)
        # idx.append(label[1])

        split_npembed=np.split(npembed,npembed.shape[1],axis=1)
        split_npembed = [embed.reshape((4, 128)) for embed in split_npembed]
        X.extend(split_npembed)
        idx.append(label)


        # # compute loss
        # loss, loss_debug = criterion(embedding)
        # loss_meter.update(loss.item(), embedding[0].size(0))
        # loss_meter_npid1.update(loss_debug[0].item(), embedding[0].size(0))
        # loss_meter_npid2.update(loss_debug[1].item(), embedding[0].size(0))
        # loss_meter_cmc1.update(loss_debug[2].item(), embedding[0].size(0))
        # loss_meter_cmc2.update(loss_debug[3].item(), embedding[0].size(0))

        # # compute gradient and do SGD step during training
        # if phase == 'train':
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        # measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()

    #     # print to terminal and tensorboard
    #     step = epoch * len(loader) + i
    #     if (i+1) % cfg['print_freq'] == 0 or i == 0 or i+1 == len(loader):
    #         progress.display(i+1)

    # # Sync metrics across all GPUs and print final averages
    # if args.multiprocessing_distributed:
    #     progress.synchronize_meters(args.gpu)
    #     progress.display(len(loader)*args.world_size)

    #if tb_writter is not None:
    #    for meter in progress.meters:
    #        tb_writter.add_scalar('{}-epoch/{}'.format(phase, meter.name), meter.avg, epoch)


if __name__ == '__main__':
    main()
