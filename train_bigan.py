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

import argparse
import numpy as np
import dataproc_v2 as dp
from torchvision import transforms, utils
import torch.nn as nn
import torch
import torch.optim as optim
import time
import logging
from utils import AverageMeter, metric_pytorch, show_tensor_img, set_logger, vis_ms, save_checkpoint
from tensorboardX import SummaryWriter
import os
from model_bigan_v6 import Generator, Discriminator, Encoder
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='experiment tag')
parser.add_argument('-t', '--train_batch_size', type=int, default=64)
parser.add_argument('-e', '--n_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0002) # learning rate
parser.add_argument('--beta1', type=float, default=0.5) # beta for Adam
parser.add_argument('--wd', type=float, default=1e-5) # weight decay
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--csv_train', type=str, default='path to training data csv')
parser.add_argument('--data_root_dir', type=str, default='root dir of npy format image data')
parser.add_argument('--suffix_pre', type=str, default='planet_pre')
parser.add_argument('--suffix_post', type=str, default='planet_post')
parser.add_argument('--print_freq', type=int, default=5)

show_batch = False

def main(args):

    log_dir = "../logs/logs_{}".format(args.version)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    model_dir = "../logs/models_{}".format(args.version)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    set_logger(os.path.join(model_dir, 'train.log'))
    writer = SummaryWriter(log_dir)

    logging.info("**************Auto-Encoder****************")
    logging.info('csv_train: {}'.format(args.csv_train))
    logging.info('data root directory: {}'.format(args.data_root_dir))
    logging.info('lr: {}'.format(args.lr))
    logging.info('beta1: {}'.format(args.beta1))
    logging.info('epochs: {}'.format(args.n_epochs))
    logging.info('batch size: {}'.format(args.train_batch_size))


    # train set
    # image transforms
    rotation_angle = (0, 90, 180, 270)
    transform_trn = transforms.Compose([
        dp.RandomFlip(),
        dp.RandomRotate(starting_angle=rotation_angle, perturb_angle = 0, prob=0.5),
        dp.ToTensor(),
        ])

    trainset = dp.PatchDataset(csv_file=args.csv_train,
                               root_dir=args.data_root_dir,
                               transform=transform_trn,
                               suffix_pre=args.suffix_pre,
                               suffix_post=args.suffix_post)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    pdb.set_trace()
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netE = Encoder().to(device)

    #criterion = torch.nn.L1Loss()
    criterion = nn.BCELoss().to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.wd)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.wd)
    optimizerE = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.wd)

    #fixed_noise = torch.randn(14, 32, 1, 1, device=device)

    t0 = time.time()
    for ep in range(args.n_epochs):
        print('Epoch [{}/{}]'.format(ep + 1, args.n_epochs))
        t1 = time.time()
        train_net(trainloader, netD, netG, netE, optimizerD, optimizerG, optimizerE, criterion, ep, writer, args.print_freq, device)
        t2 = time.time()
        logging.info('Train Epoch [{}/{}] [Time: {:.4f}]'.format(
            ep+1, args.n_epochs, (t2 - t1) / 3600.0))


    logging.info('Training Done....')


def train_net(dataloader, netD, netG, netE, optimizerD, optimizerG, optimizerE, criterion, epoch, writer, print_freq, device):

    print("Training...")
    netD.train()
    netG.train()
    netE.train()

    REAL_LB = 1
    FAKE_LB = 0

    epoch_loss = AverageMeter()
    n_batchs = len(dataloader)

    w_e, w_g, w_d  = (0.4, 0.4, 0.2)

    for i, batched_data in enumerate(dataloader):

        patch_pre = batched_data['patch_pre'].to(device=device, dtype=torch.float32) # [Batch, 4, h, w]
        patch_post = batched_data['patch_post'].to(device=device, dtype=torch.float32)
        #label = batched_data['label'].to(device=device, dtype=torch.float32)
        batch_size = patch_pre.shape[0]

        x = torch.cat((patch_pre, patch_post), dim=0)  # stack across batch dimension, [2*Batch, 4, h, w]
        del patch_pre
        del patch_post

        ######################################
        # Update E network
        ######################################
        netD.zero_grad()
        netE.zero_grad()
        en_x = netE(x)
        ## Train with all real batch
        output = netD(x, en_x).view(-1)
        label = torch.full((batch_size*2,), REAL_LB, device=device)
        errDE_real = criterion(output, label)
        # update E
        errDE_real.backward(retain_graph=True)
        optimizerE.step()

        D_x_en_x = output.mean().item()

        ######################################
        # Update D network
        ######################################
        ## Train with all fake batch
        #z = torch.randn(batch_size*2, 32, 1, 1, device=device)
        z = torch.rand(batch_size*2, 32, 1, 1, device=device) * 2 - 1 # uniform distribution [-1, 1]
        # Generate fake image batch with G
        gz = netG(z)
        label.fill_(FAKE_LB)
        output = netD(gz, z).view(-1)
        errDG_fake = criterion(output, label)
        # update D
        errDG_fake.backward(retain_graph=True)
        optimizerD.step()
        # loss on both real and fake batch
        errD = errDE_real + errDG_fake
        # discrimination on fake batches before updating D
        D_G_z1 = output.mean().item()

        ######################################
        # Update G network
        ######################################
        netG.zero_grad()
        label.fill_(REAL_LB)
        output = netD(gz, z).view(-1)
        errG = criterion(output, label)
        # update G
        errG.backward()
        optimizerG.step()
        # discrimination on fake batches after updating D
        D_G_z2 = output.mean().item()

        if i % (n_batchs//print_freq + 1)  == 0:
            print('[%d][%d/%d]\t '
                  'Loss_D: %.4f\t Loss_G: %.4f\t D(x): %.4f\t D[G(z)]: %.4f / %.4f' %
                  (epoch+1, i, n_batchs,
                   errD, errG, D_x_en_x, D_G_z1, D_G_z2))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)