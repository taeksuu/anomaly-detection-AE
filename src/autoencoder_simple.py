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

from Dataset import MVTecADDataset, Rescale, RandomCrop, RandomTranslation, RandomRotation, ToTensor


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
                
        return x
      
      
directory = "D:\mvtec_anomaly_detection"

transform_texture = [RandomCrop(256), ToTensor()]
transform_object = [RandomTranslation(40), Rescale(256), ToTensor()]

train_grid = MVTecADDataset(directory, category="grid", mode="train", transform=transforms.Compose(transform_texture))
validate_grid = MVTecADDataset(directory, category="grid", mode="validate", transform=transforms.Compose(transform_texture))
test_grid = MVTecADDataset(directory, category="grid", mode="test", transform=transforms.Compose(transform_texture))

train_hazelnut = MVTecADDataset(directory, category="hazelnut", mode="train", transform=transforms.Compose(transform_object))
validate_hazelnut = MVTecADDataset(directory, category="hazelnut", mode="validate", transform=transforms.Compose(transform_object))
test_hazelnut = MVTecADDataset(directory, category="hazelnut", mode="test", transform=transforms.Compose([Rescale(256), ToTensor()]))


train_grid_dataloader = DataLoader(train_grid, batch_size=128, shuffle=True, num_workers=0)
validate_grid_dataloader = DataLoader(validate_grid, batch_size=128, shuffle=True, num_workers=0)
test_grid_dataloader = DataLoader(test_grid, batch_size=128, shuffle=True, num_workers=0)

train_hazelnut_dataloader = DataLoader(train_hazelnut, batch_size=16, shuffle=True, num_workers=0)
validate_hazelnut_dataloader = DataLoader(validate_hazelnut, batch_size=16, shuffle=True, num_workers=0)
test_hazelnut_dataloader = DataLoader(test_hazelnut, batch_size=16, shuffle=True, num_workers=0)


num_epochs = 90
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print(device)
model_conv = ConvAutoEncoder().to(device)
loss_fn_conv = nn.MSELoss().to(device)
optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.001)

for i in range(num_epochs):
    train_loss = 0.0
    for j, train_data in enumerate(train_hazelnut_dataloader):
        x = train_data['image'].to(device)
        x = x.float().to(device)
        
        optimizer_conv.zero_grad()
        output = model_conv(x)
        loss = loss_fn_conv(output, x)
        loss.backward()
        optimizer_conv.step()
        train_loss += loss.item() * x.size(0)
        print('{}th batch learned'.format(j))
    train_loss = train_loss / len(train_hazelnut_dataloader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        i, 
        train_loss
        ))
    torch.save(model_conv.state_dict(), "C:/Users/Taeksoo Kim/Anomaly Detection/conv_model_1/epoch_{}.pt".format(i+1))
    
    
for i, test_data in enumerate(test_hazelnut_dataloader):
  if i == 0:
      fig = plt.figure(figsize=(8, 4))
      ax1 = fig.add_subplot(1, 2, 1)
      plt.imshow(test_data['image'][0].numpy().transpose((1, 2, 0)))

      x = test_data['image'].to(device)
      out = model_conv(x.float())
      ax2 = fig.add_subplot(1, 2, 2)
      plt.imshow(out[0].cpu().detach().numpy().transpose((1, 2, 0)))
      print(out.shape)
      break
