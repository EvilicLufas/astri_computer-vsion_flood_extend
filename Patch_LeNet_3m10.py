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
import torch.nn.functional as F


class Standard_AutoEncoder(nn.Module):

    def __init__(self, in_ch=4, rep_dims=64):
        super(Standard_AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, bias=True),
            nn.BatchNorm2d(64, eps=1e-04, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, bias=True),
            nn.BatchNorm2d(128, eps=1e-04, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, rep_dims, 1, bias=True),
            #nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(rep_dims, 128, 1, bias=True),
            nn.BatchNorm2d(128, eps=1e-04, affine=True),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(128, 64, 3, bias=True),
            nn.BatchNorm2d(64, eps=1e-04, affine=True),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(64, in_ch, 3, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.decoder(h)

        return z


# 0319_ae
class AutoEncoder(nn.Module):

    def __init__(self, in_ch=4):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, bias=False),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 32, 1, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(64, 32, 3, bias=False),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, in_ch, 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.decoder(h)

        return z


class DeepSVDD(nn.Module):

    def __init__(self, in_ch=4):
        super(DeepSVDD, self).__init__()

        # must match the encoder in AutoEncoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, bias=False),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 32, 1, bias=False)
        )

    def forward(self, x):

        h = self.encoder(x).view(x.size(0), -1)
        return h


class Patch_LeNet(nn.Module):

    def __init__(self, in_ch=4):
        super(Patch_LeNet, self).__init__()

        #self.rep_dim = 10
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (match the AE network)
        self.conv1 = nn.Conv2d(in_ch, 8, 3, bias=True, padding=0)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=True)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=True, padding=0)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=True)

        # fully connection for decision
        self.fc1 = nn.Linear(16 * 2, 16, bias=True)
        self.fc2 = nn.Linear(16, 1, bias=True)

    def forward(self, x1, x2):

        # Encoder
        # pre
        x1 = self.conv1(x1) # [8, 8, 8]
        x1 = self.pool(F.leaky_relu(self.bn1(x1))) # [8, 4, 4]
        x1 = self.conv2(x1) # [16, 2, 2]
        x1 = self.pool(F.leaky_relu(self.bn2(x1))) # [16, 1, 1]
        x1 = x1.view(x1.size(0), -1) # [16]

        # post
        x2 = self.conv1(x2) # [8, 8, 8]
        x2 = self.pool(F.leaky_relu(self.bn1(x2))) # [8, 4, 4]
        x2 = self.conv2(x2) # [16, 2, 2]
        x2 = self.pool(F.leaky_relu(self.bn2(x2))) # [16, 1, 1]
        x2 = x2.view(x2.size(0), -1) # [16]

        # decision
        x = torch.cat((x1, x2), dim=1) # [2*16 = 32]
        x = F.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x


class Patch_LeNet_AE_Encoder(nn.Module):

    def __init__(self, in_ch=4):
        super(Patch_LeNet_AE_Encoder, self).__init__()

        #self.rep_dim = 10
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv1 = nn.Conv2d(in_ch, 8, 3, bias=False, padding=0)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False, padding=0)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        #self.fc1 = nn.Linear(4 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        # Encoder
        x = self.conv1(x) # [8, 8, 8]
        x = self.pool(F.leaky_relu(self.bn1(x))) # [8, 4, 4]
        x = self.conv2(x) # [16, 2, 2]
        x = self.pool(F.leaky_relu(self.bn2(x))) # [16, 1, 1]
        x = x.view(x.size(0), -1) # [16]
        #x = self.fc1(x)

        return x

