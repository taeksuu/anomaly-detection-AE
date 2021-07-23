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


class AutoEncoderSeq(torch.nn.Module):
    def __init__(self, color_mode, directory, loss_fn, latent_space_dim=128, batch_size=128, verbose=True):
        super(AutoEncoderSeq, self).__init__()
        
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

    
directory = "D:\mvtec_anomaly_detection"

transform_texture = [RandomCrop(256), ToTensor()]
transform_object = [RandomTranslation(40), RandomRotation(20), Rescale(256), ToTensor()]

train_grid = MVTecADDataset(directory, category="grid", mode="train", transform=transforms.Compose(transform_texture))
validate_grid = MVTecADDataset(directory, category="grid", mode="validate", transform=transforms.Compose(transform_texture))
test_grid = MVTecADDataset(directory, category="grid", mode="test", transform=transforms.Compose(transform_texture))

train_hazelnut_list = [MVTecADDataset(directory, category="hazelnut", mode="train", transform=transforms.Compose(transform_object)) for i in range(20)]
train_hazelnut = torch.utils.data.ConcatDataset(train_hazelnut_list)


train_grid_dataloader = DataLoader(train_grid, batch_size=128, shuffle=True, num_workers=0)
validate_grid_dataloader = DataLoader(validate_grid, batch_size=128, shuffle=True, num_workers=0)
test_grid_dataloader = DataLoader(test_grid, batch_size=128, shuffle=True, num_workers=0)

train_hazelnut_dataloader = DataLoader(train_hazelnut, batch_size=128, shuffle=True, num_workers=0)
validate_hazelnut_dataloader = DataLoader(validate_hazelnut, batch_size=128, shuffle=True, num_workers=0)
test_hazelnut_dataloader = DataLoader(test_hazelnut, batch_size=128, shuffle=True, num_workers=0)


num_epochs = 200

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print(device)
model = AutoEncoderSeq(color_mode="rgb", directory=None, loss_fn="L2", latent_space_dim=128, batch_size=128, verbose=True).to(device)
loss_fn = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,threshold=0.1, patience=1, mode='min') 


torch.backends.cudnn.benchmark = True
model.train()

for i in range(num_epochs):
    train_loss = 0.0
    for j, train_data in enumerate(train_hazelnut_dataloader):
        x = train_data['image'].to(device)
        x = x.float().to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        print('{}th batch learned'.format(j))
    #scheduler.step(loss)
    train_loss = train_loss / len(train_hazelnut_dataloader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        i, 
        train_loss
        ))
    if (i+1) % 10 == 0: torch.save(model.state_dict(), "D:/model/epoch_{}.pt".format(i+1))
        
        
for i, test_data in enumerate(test_hazelnut_dataloader):
    if i == 0:
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        plt.imshow(test_data['image'][0].numpy().transpose((1, 2, 0)))
        
        x = test_data['image'].to(device)
        out = model(x.float())
        ax2 = fig.add_subplot(1, 2, 2)
        plt.imshow(out[0].cpu().detach().numpy().transpose((1, 2, 0)))
        print(out.shape)
        break
