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

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import os
import shutil
from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def resize_image(img, new_size, interpolation=cv2.INTER_LINEAR):
  # resize an image into new_size (w * h) using specified interpolation
  # opencv has a weird rounding issue & this is a hacky fix
  # ref: https://github.com/opencv/opencv/issues/9096
  mapping_dict = {cv2.INTER_NEAREST: Image.NEAREST}
  if interpolation in mapping_dict:
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(new_size,
                             resample=mapping_dict[interpolation])
    img = np.array(pil_img)
  else:
    img = cv2.resize(img, new_size,
                     interpolation=interpolation)
  return img


def vis_ms(img_ms, r, g, b):
    """

    :param img_ms: tensor images [B, C, H, W]
    :param r: int
    :param g: int
    :param b: int
    :return:
    """
    # extract rgb bands from multispectral image
    img_ms_subspec = torch.cat((img_ms[:, r].unsqueeze(1), img_ms[:, g].unsqueeze(1), img_ms[:, b].unsqueeze(1)), dim=1)
    return img_ms_subspec


SMOOTH = 1e-6

def metric_pytorch(y_pred, y_true):
    """
    :param y_pred: (tensor) 1D
    :param y_true: (tensor) 1D
    :return: precision, recall, f1, accuracy
    """
    assert y_pred.shape == y_true.shape

    n_sample = y_true.numel()
    acc = (y_pred == y_true).sum().float() / n_sample

    TP = (y_pred & y_true).sum().float()  # true positive
    FP = y_pred.sum().float() - TP  # false positive
    FN = y_true.sum().float() - TP  # false negative
    TN = (y_pred == y_true).sum().float() - TP

    precision = (TP + SMOOTH) / (TP + FP + SMOOTH)
    recall = (TP + SMOOTH) / (TP + FN + SMOOTH)
    f1 = 2.0 / (1.0 / precision + 1.0 / recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    if torch.abs(acc - accuracy).sum() > 0.0001:
        print('acc computation wrong')
        print('acc', acc)
        print('accuracy', accuracy)
        return None

    return precision, recall, f1, acc


def save_tensor_imgs(imgs, root_dir, ids, mean=None, std=None, scale=1):
    """

    :param imgs: tensor, [B, C, H, W]
    :return:
    """

    batch = imgs.shape[0]
    for b in range(batch):
        img = imgs[b]
        if mean is not None and std is not None:
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m).mul_(scale)
        else:
            for t in img:
                t.mul_(scale)
        img_np = img.detach().cpu().numpy().transpose((1, 2, 0))
        np.save(os.path.join(root_dir, '{}_sr_ms.npy'.format(ids[b])), img_np)



def show_tensor_img(img):
    """
    show tensor image
    :param img: (tensor) [C, H, W]
    :return:
    """
    plt.figure()
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, root_dir):
    filename_chk = os.path.join(root_dir, 'checkpoint.pth.tar')
    torch.save(state, filename_chk)
    if is_best:
        filename_modelbest = os.path.join(root_dir, 'model_best.pth.tar')
        shutil.copyfile(filename_chk, filename_modelbest)
