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
from PIL import Image
import numpy as np
from sklearn import preprocessing as skp

def load_image(path):
  # load an image
  img = cv2.imread(path)
  img = img[:, :, ::-1]  # BGR -> RGB
  return img

def save_image(path, img):
  img = img.copy()[:,:,::-1]
  return cv2.imwrite(path, img)

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


def RescaleImgIntesity(img, out_range):
    
    minmax = skp.MinMaxScaler(out_range)
    H, W, C = img.shape
    img = np.reshape(img, (H*W, C)) # [H, W, C] -> [H*W, C]
    img = minmax.fit_transform(img)
    img = np.reshape(img, (H, W, C)) # get back to image 

    return img

