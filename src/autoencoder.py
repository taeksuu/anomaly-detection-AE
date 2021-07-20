from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

plt.ion()

from Dataset import MVTecADDataset, ToTensor, visualize_batch


class AutoEncoder(torch.nn.Module):
    def __init__(self, color_mode, directory, loss_fn, latent_space_dim=128, batch_size=128, verbose=True):
        super(AutoEncoder, self).__init__()
        
        self.color_mode = color_mode
        self.directory = directory
        self.loss_fn = loss_fn
        self.latent_space_dim = latent_space_dim
        self.batch_size = batch_size
        self.verbose = verbose
        
        if color_mode == "rgb":
            channels = 3
        else: channels = 1
        
        #encoder 
        self.encoder = nn.Sequential(
            #Conv1
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv3
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv5
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv6
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv7
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv8
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv9
            nn.Conv2d(in_channels=32, out_channels=self.latent_space_dim, kernel_size=8, stride=1, padding=0)
        )
        
        #decoder
        self.decoder = nn.Sequential(
            #Conv9 reversed
            nn.ConvTranspose2d(in_channels=self.latent_space_dim, out_channels=32, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv8 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv7 reversed
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv6 reversed
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv5 reversed
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv4 reversed
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv3 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv2 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            #Conv1 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1),
        )
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
