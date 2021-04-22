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
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='experiment tag')
parser.add_argument('-t', '--train_batch_size', type=int, default=16)
parser.add_argument('-v', '--valid_batch_size', type=int, default=16)
parser.add_argument('-e', '--n_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01) # learning rate
parser.add_argument('--wd', type=float, default=1e-5) # weight decay
parser.add_argument('--beta1', type=float, default=0.5) # weight decay
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--csv_train', type=str, default='path to training data csv')
parser.add_argument('--csv_valid', type=str, default='path to validation data csv')
parser.add_argument('--data_root_dir', type=str, default='root dir of npy format image data')
parser.add_argument('--suffix_pre', type=str, default='planet_pre')
parser.add_argument('--suffix_post', type=str, default='planet_post')
parser.add_argument('--print_freq', type=int, default=5)
parser.add_argument('--loss', type=str, default='l1')


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

    logging.info("**************Auto-Encoder: Training****************")
    logging.info('version: {}'.format(args.version))
    logging.info('csv_train: {}'.format(args.csv_train))
    logging.info('csv_valid: {}'.format(args.csv_valid))
    logging.info('data root directory: {}'.format(args.data_root_dir))
    logging.info('learning rate: {}'.format(args.lr))
    logging.info('beta1: {}'.format(args.beta1))
    logging.info('epochs: {}'.format(args.n_epochs))
    logging.info('train / valid batch size: {}'.format(args.train_batch_size, args.valid_batch_size))
    logging.info('loss: {}'.format(args.loss))

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

    net = Model().to(device)

    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()
    logging.info(criterion)
    #criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    ae_min_loss = 1e06 # initialize valid loss to a large number
    if args.resume:
        checkpoint = torch.load("{}/model_best.pth.tar".format(model_dir), map_location=device)
        start_epoch = checkpoint['epoch']
        ae_min_loss = checkpoint['min_loss']
        net.load_state_dict(checkpoint['net_state_dict'])
        logging.info("resumed checkpoint at epoch {} with min loss {:.4e}"
                  .format(start_epoch, ae_min_loss))

    t0 = time.time()
    for ep in range(args.n_epochs):
        print('Epoch [{}/{}]'.format(ep + 1, args.n_epochs))
        t1 = time.time()
        loss_train = train_net(trainloader, net, optimizer, criterion, ep, writer, args.print_freq, device)
        t2 = time.time()
        writer.add_scalars('training/Loss', {"train": loss_train}, ep + 1)
        logging.info('Train Epoch [{}/{}] [Time: {:.4f}] [Loss: {:.4e}]'.format(
            ep+1, args.n_epochs, (t2 - t1) / 3600.0, loss_train))

        loss_valid = valid_net(validloader, net, criterion, ep, writer, args.print_freq, device)
        t3 = time.time()
        writer.add_scalars('training/Loss', {"valid": loss_valid}, ep + 1)
        logging.info('Valid Epoch [{}/{}] [Time: {:.4f}] [Loss: {:.4e}]'.format(
            ep+1, args.n_epochs, (t3 - t2) / 3600.0, loss_valid))

        logging.info('Time spent total at [{}/{}]: {:.4f}'.format(ep + 1, args.n_epochs, (t3 - t0) / 3600.0))

        # remember best prec@1 and save checkpoint
        is_best = loss_valid < ae_min_loss
        ae_min_loss = min(loss_valid, ae_min_loss)
        scheduler.step(loss_valid)  # reschedule learning rate
        save_checkpoint({
            'epoch': ep + 1,
            'net_state_dict': net.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': ae_min_loss,
        }, is_best, root_dir=model_dir)

    logging.info('Training Done....')


def train_net(dataloader, net, optimizer, criterion, epoch, writer, print_freq=10, device='cpu'):

    net.train()
    print("Training...")

    epoch_loss = AverageMeter()
    n_batches = len(dataloader)
    #pdb.set_trace()
    for i, batched_data in enumerate(dataloader):

        patch_pre = batched_data['patch_pre'].to(device=device, dtype=torch.float32) # [Batch, 4, h, w]
        patch_post = batched_data['patch_post'].to(device=device, dtype=torch.float32)
        #label = batched_data['label'].to(device=device, dtype=torch.float32)
        batch_size = patch_post.shape[0]

        # AutoEncoder
        patch_pre_r = net(patch_pre)
        patch_post_r = net(patch_post)
        # reconstruction loss
        loss_pre = criterion(patch_pre, patch_pre_r)
        loss_post = criterion(patch_post, patch_post_r)
        loss = loss_pre + loss_post
        epoch_loss.update(loss.item(), batch_size)

        # accuracy
        writer.add_scalar('training/train_loss', loss.item(), epoch*n_batches+i)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (n_batches//print_freq + 1)  == 0:
            logging.info('[%d][%d/%d]\t loss: %.4e' %
                  (epoch+1, i, n_batches, epoch_loss.avg))

    return epoch_loss.avg


def valid_net(dataloader, net, criterion, epoch, writer, print_freq=10, device='cpu'):

    net.eval()
    print("Validation...")

    epoch_loss = AverageMeter()
    n_batches = len(dataloader)

    for i, batched_data in enumerate(dataloader):

        patch_pre = batched_data['patch_pre'].to(device=device, dtype=torch.float32) # [Batch, 4, h, w]
        patch_post = batched_data['patch_post'].to(device=device, dtype=torch.float32)
        #label = batched_data['label'].to(device=device, dtype=torch.float32)
        batch_size = patch_post.shape[0]

        # AutoEncoder
        patch_pre_r = net(patch_pre)
        patch_post_r = net(patch_post)
        # reconstruction loss
        loss_pre = criterion(patch_pre, patch_pre_r)
        loss_post = criterion(patch_post, patch_post_r)
        loss = loss_pre + loss_post
        epoch_loss.update(loss.item(), batch_size)

        # accuracy
        writer.add_scalar('training/valid_loss', loss.item(), epoch*n_batches+i)

        if i % (n_batches//print_freq + 1)  == 0:
            logging.info('[%d][%d/%d]\t loss: %.4e' %
                  (epoch+1, i, n_batches, epoch_loss.avg))

    return epoch_loss.avg


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)