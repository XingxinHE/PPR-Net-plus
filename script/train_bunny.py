import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from pprnet import ROOT_DIR
from torchvision import transforms
from torch.utils.data import DataLoader

from pprnet.pprnet import PPRNet, load_checkpoint, save_checkpoint
from pprnet.object_type import ObjectType

from pprnet.utils.train_helper import BNMomentumScheduler, OptimizerLRScheduler, SimpleLogger
from pprnet.data.IPA_pose_dataset import IPAPoseDataset
from pprnet.data.pointcloud_transforms import PointCloudShuffle, ToTensor
import warp as wp
from time import perf_counter

def train_one_epoch(loader):
    logger.log_string('--------------------')
    net.train() # set model to training mode
    
    total_batch = 0
    total_seen = 0
    loss_sum = 0
    rot_loss_sum = 0
    vs_loss_sum = 0
    trans_loss_sum = 0
    dist_mean_sum = 0

    for batch_idx, batch_samples in enumerate(loader):
        total_batch += 1
        labels = {
            'rot_label':batch_samples['rot_label'].to(device),
            'trans_label':batch_samples['trans_label'].to(device),
            'vis_label':batch_samples['vis_label'].to(device)
        }
        inputs = {
            'point_clouds': batch_samples['point_clouds'].to(device),
            'labels': labels
        }

        # Forward pass
        optimizer.zero_grad()
        pred_results, losses = net(inputs)
        losses['total'].backward()
        optimizer.step()

        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += losses['total'].item()
        rot_loss_sum += losses['rot_head'].item()
        trans_loss_sum += losses['trans_head'].item()
        vs_loss_sum += losses['vis_head'].item()

        dist_mean = torch.mean(torch.norm(pred_results[0].view(-1,3)-labels['trans_label'].view(-1,3), dim=1)).item()
        dist_mean_sum += dist_mean

        if batch_idx % DISPLAY_BATCH_STEP == 0 and batch_idx!= 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,len(loader)))
            tran_loss_cur = trans_loss_sum/(batch_idx+1)
            rot_loss_cur = rot_loss_sum/(batch_idx+1)
            dist_mean_cur = dist_mean_sum/(batch_idx+1)
            vs_loss_cur = vs_loss_sum/(batch_idx+1)
            print('trans_loss: %f\trot_loss: %f\tvs_loss: %f\tmean_dist: %f'%(tran_loss_cur,rot_loss_cur, vs_loss_cur,dist_mean_cur))

    logger.log_string('train translation loss: %f' % (trans_loss_sum / float(total_batch)))
    logger.log_string('train rotation loss: %f' % (rot_loss_sum / float(total_batch)))
    logger.log_string('train vis loss: %f' % (vs_loss_sum / float(total_batch)))
    logger.log_string('train dist: %f' % (dist_mean_sum / float(total_batch)))

def eval_one_epoch(loader):
    logger.log_string('--------------------')
    net.eval() # set model to eval mode
    
    total_batch = 0
    total_seen = 0
    loss_sum = 0
    rot_loss_sum = 0
    vs_loss_sum = 0
    trans_loss_sum = 0
    dist_mean_sum = 0

    for batch_idx, batch_samples in enumerate(loader):
        total_batch += 1
        labels = {
            'rot_label':batch_samples['rot_label'].to(device),
            'trans_label':batch_samples['trans_label'].to(device),
            'vis_label':batch_samples['vis_label'].to(device)
        }
        inputs = {
            'point_clouds': batch_samples['point_clouds'].to(device),
            'labels': labels
        }

        # Forward pass
        with torch.no_grad():
            pred_results, losses = net(inputs)

            total_seen += (BATCH_SIZE*NUM_POINT)
            loss_sum += losses['total'].item()
            rot_loss_sum += losses['rot_head'].item()
            trans_loss_sum += losses['trans_head'].item()
            vs_loss_sum += losses['vis_head'].item()

            dist_mean = torch.mean(torch.norm(pred_results[0].view(-1,3)-labels['trans_label'].view(-1,3), dim=1)).item()
            dist_mean_sum += dist_mean

    logger.log_string('eval translation loss: %f' % (trans_loss_sum / float(total_batch)))
    logger.log_string('eval rotation loss: %f' % (rot_loss_sum / float(total_batch)))
    logger.log_string('eval vis loss: %f' % (vs_loss_sum / float(total_batch)))
    logger.log_string('eval dist: %f' % (dist_mean_sum / float(total_batch)))
    return (loss_sum / float(total_batch))

def train(start_epoch):
    min_loss = 1e10
    train_dataset = None
    for epoch in range(start_epoch, MAX_EPOCH):
        # loading dataset
        if epoch%TRAIN_DATA_HOLD_EPOCH == 0 or train_dataset is None:
            cid = int(epoch/TRAIN_DATA_HOLD_EPOCH) % len(TRAIN_CYCLE_RANGES)
            print('Loading train dataset...')
            train_dataset = IPAPoseDataset(DATASET_DIR, TRAIN_CYCLE_RANGES[cid], TRAIN_SCENE_RANGE, transforms=transforms)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            print('Train dataset loaded, train point cloud size:', train_dataset.dataset['data'].shape)

        logger.log_string('************** EPOCH %03d **************' % (epoch))
        logger.log_string(str(datetime.now()))

        bnm_scheduler.step(epoch) # decay BN momentum
        lr_scheduler.step(epoch)

        logger.log_string('Current learning rate: %s'%str(lr_scheduler.get_optimizer_lr()))
        logger.log_string('Current BN decay momentum: %f'%(bnm_scheduler.get_bn_momentum(epoch)))

        start = perf_counter()
        loss = train_one_epoch(train_loader)
        print('Training Time:', (perf_counter() - start))
        if epoch%EVAL_STAP == 0:
            loss = eval_one_epoch(test_loader)
            if loss < min_loss:
                min_loss = loss
            save_checkpoint(os.path.join(log_dir, 'checkpoint.tar'), epoch, net, optimizer, loss)
            logger.log_string("Model saved in file: %s" % os.path.join(log_dir, 'checkpoint.tar'))
        print("\n")

if __name__=='__main__':
    # -----------------------1. GLOBAL SETTINGS START-----------------------
    BATCH_SIZE = 8
    NUM_POINT = 2**14
    MAX_EPOCH = 5000
    # log setting
    PROJECT_NAME = "ppr"
    LOG_NAME = 'log0_batch8_scale3_test_log'
    CHECKPOINT_PATH = None  # str or None, if is not None, model will continue to train from CHECKPOINT_PATH
    # lr decay
    BASE_LEARNING_RATE = 0.001
    LR_DECAY_RATE = 0.7
    MIN_LR = 1e-6
    LR_DECAY_STEP = 80
    LR_LAMBDA = lambda epoch: max(BASE_LEARNING_RATE * LR_DECAY_RATE ** (int(epoch / LR_DECAY_STEP)), MIN_LR)
    # bn decay
    # Decay Batchnorm momentum from 0.5 to 0.999
    # note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MIN = 0.001
    BN_DECAY_STEP = LR_DECAY_STEP
    BN_DECAY_RATE = 0.5
    BN_LAMBDA = lambda epoch: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(epoch / BN_DECAY_STEP)), BN_MOMENTUM_MIN)
    # dataset settings
    DATASET_DIR = f'E:\\h5_dataset\\bunny\\'
    TRAIN_DATA_HOLD_EPOCH = 2
    TRAIN_CYCLE_RANGES = [[id, id + 1] for id in np.arange(1, 230)]
    TRAIN_SCENE_RANGE = [3, 81]
    TEST_CYCLE_RANGE = [229, 230]
    TEST_SCENE_RANGE = TRAIN_SCENE_RANGE
    # train setting
    EVAL_STAP = 1  # run evaluation after training EVAL_STAP epoch(s)
    DISPLAY_BATCH_STEP = 100
    # -----------------------GLOBAL SETTINGS END-----------------------

    # -----------------------2. BUILD NETWORK AND OTHERS FROM GLOBAL SETTINGS-----------------------
    # build logger
    log_dir = os.path.join(ROOT_DIR, 'logs', PROJECT_NAME, LOG_NAME)
    logger = SimpleLogger(log_dir)

    # build net and optimizer( and load if CHECKPOINT_PATH is not None )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    type_bunny = ObjectType(type_name='bunny', class_idx=0, symmetry_type='finite',
                            lambda_p=[[0.0263663, 0.0, 0.0], [0.0, 0.0338224, 0.0], [-0.0, 0.0, 0.0484393]],
                            G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

    backbone_config = {
        'npoint_per_layer': [4096, 1024, 256, 64],
        'radius_per_layer': [[10, 20, 30], [30, 45, 60], [60, 80, 120], [120, 160, 240]]
    }

    loss_weights = {
        'trans_head': 200.0 / 1000.0,  # implicit convert mm to m by deviding 1000
        'rot_head': 200.0,
        'vis_head': 50.0
    }
    wp.init()
    net = PPRNet(type_bunny, backbone_config, True, loss_weights, True)
    if torch.cuda.device_count() > 1:
        logger.log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE)
    if CHECKPOINT_PATH is not None:
        net, optimizer, start_epoch = load_checkpoint(CHECKPOINT_PATH, net, optimizer)
    else:
        start_epoch = 0

    # build scheduler
    bnm_scheduler = BNMomentumScheduler(net, bn_lambda=BN_LAMBDA, last_epoch=start_epoch - 1)
    lr_scheduler = OptimizerLRScheduler(optimizer, lr_lambda=LR_LAMBDA, last_epoch=start_epoch - 1)

    # dataset
    transforms = transforms.Compose(
        [
            PointCloudShuffle(NUM_POINT),
            ToTensor()
        ]
    )
    print('Loading test dataset')
    test_dataset = IPAPoseDataset(DATASET_DIR, TEST_CYCLE_RANGE, TRAIN_SCENE_RANGE, transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print('Test dataset loaded, test point cloud size:', test_dataset.dataset['data'].shape)
    # -----------------------END-----------------------

    train(start_epoch)
