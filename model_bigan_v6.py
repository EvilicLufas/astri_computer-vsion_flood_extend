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

class Generator(nn.Module):
    def __init__(self, in_ch=4, rep_dims=32):

        super(Generator, self).__init__()

        self.generate = nn.Sequential(
            nn.ConvTranspose2d(rep_dims, 128, 2, bias=False), # [128, 2, 2]
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, bias=False), # [64, 4, 4]
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 3, bias=False),  # [32, 6, 6]
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 3, bias=False),  # [16, 8, 8]
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, in_ch, 3, bias=True), # [in_ch, 10, 10]
            #nn.Sigmoid()
            nn.Tanh()
        )

        #for m in self.modules():
        #    if isinstance(m, nn.ConvTranspose2d):
        #        m.weight.data.normal_(0.0, 0.02)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.normal_(1.0, 0.02)
        #        m.bias.data.zero_()

    def forward(self, z):
        out = self.generate(z)
        return out


class Encoder(nn.Module):
    def __init__(self, in_ch=4, rep_dims = 32):

        super(Encoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, bias=False), # [16, 8, 8]
            nn.BatchNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 3, bias=False), # [32, 6, 6]
            nn.BatchNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, bias=False),  # [64, 4, 4]
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, bias=False),  # [128, 2, 2]
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, rep_dims, 2, bias=False), # [rep_dims, 1, 1]
            #nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.Tanh()
        )
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        m.weight.data.normal_(0.0, 0.02)
        #    elif isinstance(m, nn.BatchNorm2d):
        #       m.weight.data.normal_(1.0, 0.02)
        #        m.bias.data.zero_()

    def forward(self, x):
        out = self.encode(x)
        return out


class Discriminator(nn.Module):

    def __init__(self, in_ch=4, rep_dims=32):

        super(Discriminator, self).__init__()

        self.encode_x = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, bias=False), # [16, 8, 8]
            nn.BatchNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 3, bias=False), # [32, 6, 6]
            nn.BatchNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, bias=False),  # [64, 4, 4]
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, bias=False),  # [128, 2, 2]
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, rep_dims, 2, bias=False), # [rep_dims, 1, 1]
            #nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.Tanh()
        )

        self.discriminate = nn.Sequential(
            # input dim: [128*2, 1, 1], concatenate
            nn.Conv2d(rep_dims*2, rep_dims*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(rep_dims*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. [128, 1, 1]
            nn.Conv2d(rep_dims*2, rep_dims, 1, 1, 0, bias=False),
            nn.BatchNorm2d(rep_dims, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. [64, 1, 1]
            nn.Conv2d(rep_dims, 1, 1, 1, 0, bias=False),
            # output size. [1, 1, 1]
            nn.Sigmoid()
        )

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        m.weight.data.normal_(0.0, 0.02)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.normal_(1.0, 0.02)
        #        m.bias.data.zero_()

    def forward(self, x, z):

        code_x = self.encode_x(x)
        #out_z = self.encode_z(z)

        x_z = torch.cat((code_x, z), dim=1)
        out = self.discriminate(x_z)

        return out