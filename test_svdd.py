"""
@ Author: Bo Peng (bo.peng@wisc.edu)
@ Spatial Computing and Data Mining Lab
@ University of Wisconsin - Madison
@ Project: Microsoft AI for Earth Project
 "Self-supervised deep learning and computer vision for
 real-time large-scale high-definition flood extent mapping"
@ Citation:
B. Peng et al., “Urban Flood Mapping With Bitemporal Multispectral
Imagery Via a Self-Supervised Learning Framework,”
IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens., vol. 14, pp. 2001–2016, 2021.
"""

import numpy as np
import dataproc as dp
from torchvision import transforms, utils
import torch
import time
import logging
from utils import AverageMeter, metric_pytorch, show_tensor_img, set_logger, vis_ms, save_checkpoint
from tensorboardX import SummaryWriter
import os
from Patch_LeNet_3m10 import DeepSVDD as Model
import argparse
import pdb
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='experiment tag')
parser.add_argument('-t', '--test_batch_size', type=int, default=64)
parser.add_argument('--csv_test', type=str, default='path to testing data csv')
parser.add_argument('--data_root_dir', type=str, default='root dir of npy format image data')
parser.add_argument('--suffix_pre', type=str, default='planet_pre')
parser.add_argument('--suffix_post', type=str, default='planet_post')
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('-s', '--soft_boundary', action='store_true', default=False)


def main(args):

    model_dir = "../logs/pretrained/models_{}".format(args.version)
    if not os.path.isdir(model_dir):
        print('no pre-trained model')
        return None

    set_logger(os.path.join(model_dir, 'test.log'))

    logging.info("**************Uni-temporal SVDD - Testing****************")
    logging.info('csv_test: {}'.format(args.csv_test))
    logging.info('data root directory: {}'.format(args.data_root_dir))
    logging.info('test batch size: {}'.format(args.test_batch_size))
    logging.info('soft-boundary: {}'.format(args.soft_boundary))


    # test set
    transform_test = transforms.Compose([
        dp.ToTensor()
    ])
    testset = dp.PatchDataset(csv_file=args.csv_test,
                               root_dir=args.data_root_dir,
                               transform=transform_test,
                               suffix_pre=args.suffix_pre,
                               suffix_post=args.suffix_post)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = Model(in_ch=4).to(device)
    #net_state_dict = net.state_dict()

    checkpoint = torch.load("{}/model_best.pth.tar".format(model_dir), map_location=device)
    start_epoch = checkpoint['epoch']
    svdd_min_loss = checkpoint['min_loss']
    center = checkpoint['center']
    radius = checkpoint['radius']
    net.load_state_dict(checkpoint['net_state_dict'])
    logging.info("pretrained uni-temporal SVDD loaded checkpoint at epoch {} with min loss {:.4e}"
              .format(start_epoch, svdd_min_loss))

    #criterion = torch.nn.L1Loss()
    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.BCELoss()

    t0 = time.time()
    distance_dict = test_net(testloader, center, radius, args.soft_boundary, net, args.print_freq, device)
    t1 = time.time()

    # save patch similarity dictionary into csv
    df = pd.DataFrame(distance_dict)
    df.to_csv(os.path.join(model_dir, 'svdd_distance_aoi.csv'), index=False)

    logging.info('Time spent total : {:.4f}'.format((t1 - t0) / 3600.0))
    logging.info('Testing Done....')


def test_net(dataloader, center, radius, soft_boundary,
             net, print_freq=10, device='cpu'):

    net.eval()
    print("Testing...")
    #pdb.set_trace()

    n_batches = len(dataloader)
    distance_dict = {'row': [], 'col': [], 'distance': []}

    for i, batched_data in enumerate(dataloader):
        row = batched_data['row'].numpy().astype(int)
        col = batched_data['col'].numpy().astype(int)
        #patch_pre = batched_data['patch_pre'].to(device=device, dtype=torch.float32) # [Batch, 4, h, w]
        patch_post = batched_data['patch_post'].to(device=device, dtype=torch.float32)
        #label = batched_data['label'].to(device=device, dtype=torch.float32)
        #batch_size = patch_pre.shape[0]

        # uni-temporal AutoEncoder
        #patch_bi = torch.cat((patch_pre, patch_post), dim=1)
        code_post = net(patch_post)
        dist = torch.sum((code_post-center)**2, dim=1).sqrt()
        # distance loss
        #if soft_boundary:
        #    dist -= radius
        #    dist = torch.max(torch.zeros_like(dist), dist)
        # distance
        distance_dict['distance'].extend(dist.detach().cpu().numpy())
        distance_dict['row'].extend(row)
        distance_dict['col'].extend(col)

        if i % (n_batches//print_freq + 1)  == 0:
            logging.info('[{:d}/{:d}]'.format(i, n_batches))

    return distance_dict


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)