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

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='experiment tag')
parser.add_argument('-t', '--train_batch_size', type=int, default=64)
parser.add_argument('-v', '--valid_batch_size', type=int, default=64)
parser.add_argument('-e', '--n_epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-4) # learning rate
parser.add_argument('--wd', type=float, default=1e-6) # weight decay
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--csv_train', type=str, default='path to training data csv')
parser.add_argument('--csv_valid', type=str, default='path to validation data csv')
parser.add_argument('--data_root_dir', type=str, default='root dir of npy format image data')
parser.add_argument('--suffix_pre', type=str, default='planet_pre')
parser.add_argument('--suffix_post', type=str, default='planet_post')
parser.add_argument('--print_freq', type=int, default=5)
parser.add_argument('--dir_pretrained_ae', type=str, default='pretrained autoencoder')
parser.add_argument('-w', '--warm_up', action='store_true', default=False)
parser.add_argument('-s', '--soft_boundary', action='store_true', default=False)
parser.add_argument('--nu', type = float, default=0.08) # fraction of outliers in training data
parser.add_argument('--warm_up_n_epochs', type=int, default=10) # number of training epochs for soft-boundary Deep SVDD before radius R gets updated


def main(args):
    #pdb.set_trace()
    log_dir = "../logs/logs_{}".format(args.version)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    model_dir = "../logs/models_{}".format(args.version)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    set_logger(os.path.join(model_dir, 'train.log'))
    writer = SummaryWriter(log_dir)

    logging.info("**************SVDD****************")
    logging.info('csv_train: {}'.format(args.csv_train))
    logging.info('csv_valid: {}'.format(args.csv_valid))
    logging.info('data root directory: {}'.format(args.data_root_dir))
    logging.info('learning rate: {}'.format(args.lr))
    logging.info('beta1: {}'.format(args.beta1))
    logging.info('epochs: {}'.format(args.n_epochs))
    logging.info('nu: {}'.format(args.nu))
    logging.info('soft-boundary: {}'.format(args.soft_boundary))
    logging.info('AE warm-up: {}'.format(args.warm_up))
    logging.info('Radius warp-up epochs: {}'.format(args.warm_up_n_epochs))
    logging.info('train / valid batch size: {} / {}'.format(args.train_batch_size, args.valid_batch_size))
    logging.info('directory of pre-trained uni-temporal AutoEncoder: {}'.format(args.dir_pretrained_ae))

    # train set
    # image transforms
    rotation_angle = (0, 90, 180, 270)
    transform_trn = transforms.Compose([
        dp.RandomFlip(),
        dp.RandomRotate(starting_angle=rotation_angle, perturb_angle = 0, prob=0.5),
        dp.ToTensor()
        ])

    trainset = dp.PatchDataset(csv_file=args.csv_train,
                               root_dir=args.data_root_dir,
                               transform=transform_trn,
                               suffix_pre=args.suffix_pre,
                               suffix_post=args.suffix_post)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    # valid set, test time augmentation (TTA)
    transform_val = transforms.Compose([
        dp.ToTensor()
    ])
    validset = dp.PatchDataset(csv_file=args.csv_valid,
                               root_dir=args.data_root_dir,
                               transform=transform_val,
                               suffix_pre=args.suffix_pre,
                               suffix_post=args.suffix_post)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #pdb.set_trace()
    net = Model(in_ch=4).to(device)
    net_state_dict = net.state_dict()

    if args.warm_up:
        # load pre-trained AutoEncoder
        checkpoint = torch.load("{}/model_best.pth.tar".format(args.dir_pretrained_ae), map_location=device)
        ae_min_loss = checkpoint['min_loss']
        ae_state_dict = checkpoint['net_state_dict']
        logging.info("Loaded Uni-temporal AutoEncoder State Dict with bi_ae_min_loss :{}".format(ae_min_loss))
        # copy encoder weights from pre-trained bi_ae to encoder in bi-svdd
        logging.info('Copying encoder weights from pre-trained AE to SVDD')
        for layer in net_state_dict:
            net_state_dict[layer] = ae_state_dict[layer]
        net.load_state_dict(net_state_dict)

    center = None
    radius = torch.tensor(0.0, device=device)

    svdd_min_loss = 1e06 # initialize valid loss to a large number
    if args.resume:
        checkpoint = torch.load("{}/model_best.pth.tar".format(model_dir), map_location=device)
        start_epoch = checkpoint['epoch']
        svdd_min_loss = checkpoint['min_loss']
        center = checkpoint['center']
        radius = checkpoint['radius']
        net.load_state_dict(checkpoint['net_state_dict'])
        logging.info("resumed checkpoint at epoch {} with min loss {:.4e}"
                  .format(start_epoch, svdd_min_loss))

    # initilize center for bi-temporal SVDD
    pdb.set_trace()
    if center is None:
        center  = init_center(trainloader, net, args.print_freq, device, eps=0.1)

    #criterion = torch.nn.L1Loss()
    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    t0 = time.time()
    for ep in range(args.n_epochs):
        print('Epoch [{}/{}]'.format(ep + 1, args.n_epochs))
        t1 = time.time()
        loss_train, radius = train_net(trainloader, center, radius, args.soft_boundary, args.nu, args.warm_up_n_epochs,
                               net, optimizer, ep, writer, args.print_freq, device)
        t2 = time.time()
        logging.info('updated radius: {:.4e}'.format(radius.item()))
        writer.add_scalars('training/Loss', {"train": loss_train}, ep + 1)
        logging.info('Train Epoch [{}/{}] [Time: {:.4f}] [Loss: {:.4e}]'.format(
            ep+1, args.n_epochs, (t2 - t1) / 3600.0, loss_train))

        loss_valid = valid_net(validloader, center, radius, args.soft_boundary, args.nu,
                               net, ep, writer, args.print_freq, device)
        t3 = time.time()
        writer.add_scalars('training/Loss', {"valid": loss_valid}, ep + 1)
        logging.info('Valid Epoch [{}/{}] [Time: {:.4f}] [Loss: {:.4e}]'.format(
            ep+1, args.n_epochs, (t3 - t2) / 3600.0, loss_valid))

        logging.info('Time spent total at [{}/{}]: {:.4f}'.format(ep + 1, args.n_epochs, (t3 - t0) / 3600.0))

        # remember best prec@1 and save checkpoint
        is_best = loss_valid < svdd_min_loss
        svdd_min_loss = min(loss_valid, svdd_min_loss)
        scheduler.step(loss_valid)  # reschedule learning rate
        save_checkpoint({
            'epoch': ep + 1,
            'net_state_dict': net.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': svdd_min_loss,
            'center': center,
            'radius': radius
        }, is_best, root_dir=model_dir)

    logging.info('Training Done....')


def train_net(dataloader, center, radius, soft_boundary, nu, warm_up_n_epochs,
              net, optimizer, epoch, writer, print_freq=10, device='cpu'):

    net.train()
    print("Training...")

    epoch_loss = AverageMeter()
    n_batches = len(dataloader)

    for i, batched_data in enumerate(dataloader):

        patch_pre = batched_data['patch_pre'].to(device=device, dtype=torch.float32) # [Batch, 4, h, w]
        patch_post = batched_data['patch_post'].to(device=device, dtype=torch.float32)
        #label = batched_data['label'].to(device=device, dtype=torch.float32)
        batch_size = patch_post.shape[0]

        pdb.set_trace()

        # Uni-temporal AutoEncoder
        #patch_bi = torch.cat((patch_pre, patch_post), dim=1)
        code_post = net(patch_post)
        dist = torch.sum((code_post-center)**2, dim=1) # squared distance
        # distance loss
        if soft_boundary:
            scores = dist - radius**2
            loss = radius**2 + (1/nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update hypersphere radius R on mini-batch distances
        if soft_boundary and epoch >= warm_up_n_epochs:
            radius = torch.tensor(get_radius(dist, nu), device=device)

        # logging
        epoch_loss.update(loss.item(), batch_size)
        writer.add_scalar('training/train_loss', loss.item(), epoch*n_batches+i)
        if i % (n_batches//print_freq + 1)  == 0:
            logging.info('[{:d}][{:d}/{:d}]\t loss: {:.4e}'.format(epoch+1, i, n_batches, epoch_loss.avg))


    return epoch_loss.avg, radius


def valid_net(dataloader, center, radius, soft_boundary, nu,
              net, epoch, writer, print_freq=10, device='cpu'):

    net.eval()
    print("Validation...")

    epoch_loss = AverageMeter()
    n_batches = len(dataloader)

    for i, batched_data in enumerate(dataloader):

        #patch_pre = batched_data['patch_pre'].to(device=device, dtype=torch.float32) # [Batch, 4, h, w]
        patch_post = batched_data['patch_post'].to(device=device, dtype=torch.float32)
        #label = batched_data['label'].to(device=device, dtype=torch.float32)
        batch_size = patch_post.shape[0]

        # Uni-temporal AutoEncoder
        #patch_bi = torch.cat((patch_pre, patch_post), dim=1)
        code_post = net(patch_post)
        dist = torch.sum((code_post-center)**2, dim=1)
        # distance loss
        if soft_boundary:
            scores = dist - radius ** 2
            loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)

        epoch_loss.update(loss.item(), batch_size)
        writer.add_scalar('training/valid_loss', loss.item(), epoch*n_batches+i)
        if i % (n_batches//print_freq + 1)  == 0:
            logging.info('[{:d}][{:d}/{:d}]\t loss: {:.4e}'.format(epoch+1, i, n_batches, epoch_loss.avg))

    return epoch_loss.avg


def init_center(dataloader, net, print_freq=10, device='cpu', eps=0.1):

    logging.info('Initializing center for uni-temporal SVDD...')
    net.eval()

    n_samples = 0
    n_batches = len(dataloader)
    center = None

    with torch.no_grad():
        for i, batched_data in enumerate(dataloader):
            #patch_pre = batched_data['patch_pre'].to(device=device, dtype=torch.float32)  # [Batch, 4, h, w]
            patch_post = batched_data['patch_post'].to(device=device, dtype=torch.float32)

            #patch_bi = torch.cat((patch_pre, patch_post), dim=1)
            code_post = net(patch_post)  # [Batch, Vector Length]

            n_samples += code_post.shape[0]
            if center is None:
                center = torch.sum(code_post, dim=0, keepdim=True)
            else:
                center += torch.sum(code_post, dim=0, keepdim=True)

            if i % (n_batches // print_freq + 1) == 0:
                logging.info('[%d/%d]' % (i, n_batches))

    center /= n_samples

    # if c_i is too close to 0, set to +- eps.
    # Reason: a zero unit can be trivially matched with zero weights
    center[(abs(center) < eps) & (center < 0)] = -eps
    center[(abs(center) < eps) & (center > 0)] = eps

    logging.info('Uni-temporal SVDD center initialized.')

    return center


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().detach().cpu().numpy()), 1 - nu)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)