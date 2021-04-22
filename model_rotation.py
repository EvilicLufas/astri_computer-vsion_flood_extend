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

import torch
import torch.nn as nn


class RotNet(nn.Module):

    def __init__(self, in_ch=4, rep_dims=256, n_rots = 4):
        super(RotNet, self).__init__()

        self.encoder = nn.Sequential(
            # state: [in_ch, 10, 10]
            nn.Conv2d(in_ch, 64, 3, bias=True), # state: [64, 8, 8]
            nn.BatchNorm2d(64, eps=1e-04, affine=True),
            nn.LeakyReLU(),

            nn.MaxPool2d(2,2), # state: [64, 4, 4]

            nn.Conv2d(64, 128, 3, bias=True), # state: [128, 2, 2]
            nn.BatchNorm2d(128, eps=1e-04, affine=True),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2,2),
            nn.Conv2d(128, rep_dims, 2, bias=True), # state: [rep_dims, 1, 1]
            #nn.BatchNorm2d(32, eps=1e-04, affine=False),

            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(rep_dims, rep_dims // 2, bias=True),
            nn.BatchNorm1d(rep_dims // 2, affine=True),
            nn.LeakyReLU(),

            nn.Linear(rep_dims // 2, 100, bias=True),
            nn.BatchNorm1d(100, affine=True),
            nn.LeakyReLU(),

            nn.Linear(100, n_rots, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.encoder(x).view(x.shape[0], -1)
        z = self.classifier(h)

        return z

