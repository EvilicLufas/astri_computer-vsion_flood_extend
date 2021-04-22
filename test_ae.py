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
from Patch_LeNet_3m10 import Standard_AutoEncoder as Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='experiment tag')
parser.add_argument('-t', '--test_batch_size', type=int, default=64)
parser.add_argument('--csv_test', type=str, default='path to testing data csv')
parser.add_argument('--data_root_dir', type=str, default='root dir of npy format image data')
parser.add_argument('--suffix_pre', type=str, default='planet_pre')
parser.add_argument('--suffix_post', type=str, default='planet_post')
parser.add_argument('--print_freq', type=int, default=5)
parser.add_argument('--loss', type=str, default='l1')


def main(args):

    model_dir = "../logs/pretrained/models_{}".format(args.version)
    if not os.path.isdir(model_dir):
        print('no pre-trained model directory')
        return None

    set_logger(os.path.join(model_dir, 'test.log'))

    logging.info("**************Auto-Encoder - Testing****************")
    logging.info('version: {}'.format(args.version))
    logging.info('csv_test: {}'.format(args.csv_test))
    logging.info('data root directory: {}'.format(args.data_root_dir))
    logging.info('test batch size: {}'.format(args.test_batch_size))
    logging.info('loss: {}'.format(args.loss))

    # valid set, test time augmentation (TTA)
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

    net = Model().to(device)

    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()
    logging.info(criterion)
    #criterion = torch.nn.BCELoss()

    checkpoint = torch.load("{}/model_best.pth.tar".format(model_dir), map_location=device)
    start_epoch = checkpoint['epoch']
    ae_min_loss = checkpoint['min_loss']
    net.load_state_dict(checkpoint['net_state_dict'])
    logging.info("resumed checkpoint at epoch {} with min loss {:.4e}"
              .format(start_epoch, ae_min_loss))

    t0 = time.time()
    loss_test = test_net(testloader, net, criterion, args.print_freq, device)
    t1 = time.time()
    logging.info('Test [Time: {:.4f}] [Loss: {:.4e}]'.format(
       (t1 - t0) / 3600.0, loss_test))

    logging.info('Testing Done....')


def test_net(dataloader, net, criterion, print_freq=10, device='cpu'):

    net.eval()
    print("Testing...")

    epoch_loss = AverageMeter()
    n_batches = len(dataloader)

    for i, batched_data in enumerate(dataloader):

        patch_pre = batched_data['patch_pre'].to(device=device, dtype=torch.float32) # [Batch, 4, h, w]
        patch_post = batched_data['patch_post'].to(device=device, dtype=torch.float32)
        #label = batched_data['label'].to(device=device, dtype=torch.float32)
        batch_size = patch_pre.shape[0]

        # AutoEncoder
        patch_pre_r = net(patch_pre)
        patch_post_r = net(patch_post)

        # reconstruction loss
        loss_pre = criterion(patch_pre, patch_pre_r)
        loss_post = criterion(patch_post, patch_post_r)
        loss = loss_pre + loss_post
        epoch_loss.update(loss.item(), batch_size)

        if i % (n_batches//print_freq + 1)  == 0:
            logging.info('[%d/%d]\t loss: %.4e' % (i, n_batches, epoch_loss.avg))

    return epoch_loss.avg


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)