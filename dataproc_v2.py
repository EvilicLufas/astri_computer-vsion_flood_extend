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
import collections
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as TF
from torchvision import transforms
import pandas as pd
import numpy as np
import os

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]


class PatchDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None, is_labeled=False, suffix_pre="planet_pre", suffix_post="planet_post"):
        """
        Args:
            root_dir (string): directory with all images
            csv_file (string): csv file with image id's
            transform (callable, optional): optional transform to be applied to a sample image.
        """
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.is_labeled = is_labeled # check if the data is labeled
        self.suffix_pre = suffix_pre
        self.suffix_post = suffix_post

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]['row'].astype(int)
        col = self.df.iloc[idx]['col'].astype(int)
        if self.is_labeled:
            label = self.df.iloc[idx]['label']
            label = (label > 0.5).astype(np.uint8)

        # input: low-resolution multi-spectral, bgr-nir
        path_patch_pre = os.path.join(self.root_dir, '{}_{}_{}.npy'.format(row, col, self.suffix_pre))
        patch_pre = np.load(path_patch_pre)  # [H, W, C]
        #H, W, C = patch_pre.shape # spectral channels

        path_patch_post = os.path.join(self.root_dir, '{}_{}_{}.npy'.format(row, col, self.suffix_post))
        patch_post = np.load(path_patch_post)  # [H, W, C]

        if self.is_labeled:
            sample = {'row': row, 'col': col, 'patch_pre': patch_pre, 'patch_post': patch_post, 'label': label}
        else:
            sample = {'row': row, 'col': col, 'patch_pre': patch_pre, 'patch_post': patch_post}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        return {'root_dir': self.root_dir,
                'csv_file': self.csv_file,
                'transform': self.transform}


class RandomFlip(object):
    """flip the given numpy array randomly
     (with a probability of 0.5).
    """
    def __call__(self, sample):

        # horizontal flip
        if random.random() < 0.5:
            sample['patch_pre'] = cv2.flip(sample['patch_pre'], 1)
            sample['patch_post'] = cv2.flip(sample['patch_post'], 1)
        # vertical flip
        if random.random() < 0.5:
            sample['patch_pre'] = cv2.flip(sample['patch_pre'], 0)
            sample['patch_post'] = cv2.flip(sample['patch_post'], 0)

        return sample


class RandomRotate(object):
    """Rotate the given numpy array (around the image center) by a random degree.

    Args:
      degree_range (float): range of degree (-d ~ +d)
    """
    def __init__(self, starting_angle=(0, 90, 180, 270), perturb_angle = 0, prob=0.5, interpolations=_DEFAULT_INTERPOLATIONS):
        # starting_angle: a tuple
        self.starting_angle = starting_angle
        self.perturb_angle = perturb_angle
        self.prob = prob
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, collections.Iterable)
        self.interpolations = interpolations

    def __call__(self, sample):
        # probability of data transform
        if random.random() > self.prob:
            return sample

        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]
        # sample rotation degree
        degree = np.random.choice(self.starting_angle) + np.random.uniform(-self.perturb_angle, self.perturb_angle)
        # ignore small rotations
        if np.abs(degree) <= 1.0:
            return sample

        # lr_ms & hr_rgb & hr_ms
        for i in range(4):
            # get the max area rectangular within the rotated image
            # ref: stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
            if i==0: # lr_ms
                h, w = sample['patch_pre'].shape[:2]
                img = sample['patch_pre']
            elif i==1:
                h, w = sample['patch_post'].shape[:2]
                img = sample['patch_post']

            side_long = float(max([h, w]))
            side_short = float(min([h, w]))

            # since the solutions for angle, -angle and pi-angle are all the same,
            # it suffices to look at the first quadrant and the absolute values of sin,cos:
            sin_a = np.abs(np.sin(np.pi * degree / 180))
            cos_a = np.abs(np.cos(np.pi * degree / 180))

            if (side_short <= 2.0 * sin_a * cos_a * side_long):
                # half constrained case: two crop corners touch the longer side,
                # the other two corners are on the mid-line parallel to the longer line
                x = 0.5 * side_short
                if w >= h:
                    wr, hr = x / sin_a, x / cos_a
                else:
                    wr, hr = x / cos_a, x / sin_a
            else:
                # fully constrained case: crop touches all 4 sides
                cos_2a = cos_a * cos_a - sin_a * sin_a
                wr = (w * cos_a - h * sin_a) / cos_2a
                hr = (h * cos_a - w * sin_a) / cos_2a

            rot_mat = cv2.getRotationMatrix2D((w/2.0, h/2.0), degree, 1.0)
            rot_mat[0,2] += (wr - w)/2.0
            rot_mat[1,2] += (hr - h)/2.0

            img = cv2.warpAffine(img, rot_mat,
                (int(round(wr)), int(round(hr))), flags=interpolation)

            if i==0: # lr_ms
                sample['patch_pre'] = img
            elif i==1:
                sample['patch_post'] = img

        return sample


class ToTensor(object):
    """
    Convert numpy.ndarray image to tensor image
    """

    def __call__(self, sample):
        assert isinstance(sample['patch_pre'], np.ndarray) # [h, w, 4]
        assert isinstance(sample['patch_post'], np.ndarray) # [H, W, 3], H=4h, W=4w

        scale = 10000.0 # planet scope data, unsigned 16, reflectance [0, 1], scaled by 10000

        sample['patch_pre'] = sample['patch_pre'].astype(np.float32) / scale
        sample['patch_post'] = sample['patch_post'].astype(np.float32) / scale

        # to clear any error data out of the range (-1, 1)
        sample['patch_pre'] = (np.clip(sample['patch_pre'], 0.0, 1.0)) * 2 - 1.0
        sample['patch_post'] = (np.clip(sample['patch_post'], 0.0, 1.0)) * 2 - 1.0

        # to tensor
        sample['patch_pre'] = torch.from_numpy(sample['patch_pre'].transpose((2, 0, 1))) #[4, h, w]
        sample['patch_post'] = torch.from_numpy(sample['patch_post'].transpose((2, 0, 1)))

        return sample

    def __repr__(self):
        return "Surface Reflectance Normalized to (0, 1)"


class GlobalContrastNormalization(object):

    def __init__(self, mean_pre, mean_post, scale_pre, scale_post):
        """
        Global Contrast Normalization
        :param mean_pre: mean of the entire image
        :param mean_post:
        :param scale_pre: l1 or l2 norm / n_features
        :param scale_post:
        """
        self.normalize_pre = transforms.Normalize(mean_pre, scale_pre)
        self.normalize_post = transforms.Normalize(mean_post, scale_post)

    def __call__(self, sample):
        # band order: B, G, R, Nir
        sample['patch_pre'] = self.normalize_pre(sample['patch_pre'])
        sample['patch_post'] = self.normalize_post(sample['patch_post'])

        return sample


class MinMaxScaling(object):

    def __init__(self, min_pre, range_pre, min_post, range_post):
        self.normalize_pre = transforms.Normalize(min_pre, range_pre)
        self.normalize_post = transforms.Normalize(min_post, range_post)

    def __call__(self, sample):
        # band order: B, G, R, Nir
        sample['patch_pre'] = self.normalize_pre(sample['patch_pre'])
        sample['patch_post'] = self.normalize_post(sample['patch_post'])

        return sample


class Normalize(object):
    """
    Normalize an tensor image with mean and standard deviation.
    """
    def __init__(self, mean_pre, std_pre, mean_post, std_post):
        self.mean_pre = mean_pre
        self.std_pre = std_pre
        self.mean_post = mean_post
        self.std_post = std_post
        self.normalize_pre = transforms.Normalize(mean_pre, std_pre)
        self.normalize_post = transforms.Normalize(mean_post, std_post)

    def __call__(self, sample):

        # band order: B, G, R, Nir
        sample['patch_pre'] = self.normalize_pre(sample['patch_pre'])
        sample['patch_post'] = self.normalize_post(sample['patch_post'])

        return sample

    def __repr__(self):
        return {'mean_pre': self.mean_pre,
                'std_pre': self.std_pre,
                'mean_post': self.mean_post,
                'std_post': self.std_post}