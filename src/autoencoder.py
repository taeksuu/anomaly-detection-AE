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
    def __init__(self, color_mode, directory, latent_space_dim=128, batch_size=128, verbose=True):
        super(AutoEncoderSeq, self).__init__()
        
        self.color_mode = color_mode
        self.directory = directory
        self.latent_space_dim = latent_space_dim
        self.batch_size = batch_size
        self.verbose = verbose
        
        if color_mode == "rgb":
            channels = 3
        else: channels = 1
        
        #encoder 
        self.encoder = nn.Sequential(
            #Additional Conv: 256 * 256 * 1
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv1
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv3
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv5
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv6
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv7
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv8
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv9
            nn.Conv2d(in_channels=32, out_channels=self.latent_space_dim, kernel_size=8, stride=1, padding=0),


        )
        
        #decoder
        self.decoder = nn.Sequential(
            
            #Conv9 reversed
            nn.ConvTranspose2d(in_channels=self.latent_space_dim, out_channels=32, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv8 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv7 reversed
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv6 reversed
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv5 reversed
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            
            #Conv4 reversed
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv3 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv2 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Conv1 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            
            #Additional Conv reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1),

        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
#directory to downloaded images
directory = "C:/Users/Taeksoo Kim/Anomaly Detection/mvtec_anomaly_detection"

transform_texture = [RandomCrop(256), ToTensor()]
transform_object = [RandomTranslation(40), RandomRotation(20), Rescale(256), ToTensor()]

#create dataset
train_hazelnut_list = [MVTecADDataset(directory, category="hazelnut", mode="train", transform=transforms.Compose(transform_object)) for i in range(20)]
train_hazelnut = torch.utils.data.ConcatDataset(train_hazelnut_list)
validate_hazelnut = MVTecADDataset(directory, category="hazelnut", mode="validate", transform=transforms.Compose(transform_object))
test_hazelnut = MVTecADDataset(directory, category="hazelnut", mode="test", transform=transforms.Compose(transform_object))

#create dataloader
train_hazelnut_dataloader = DataLoader(train_hazelnut, batch_size=128, shuffle=True, num_workers=0)
validate_hazelnut_dataloader = DataLoader(validate_hazelnut, batch_size=128, shuffle=True, num_workers=0)
test_hazelnut_dataloader = DataLoader(test_hazelnut, batch_size=128, shuffle=True, num_workers=0)


def train(load_model_path=None, save_model_path=None, dataloader=None, num_epochs=100, criterion=None, optimizer=None, writer=None):
    
    model = AutoEncoderSeq(color_mode="rgb", directory=None, latent_space_dim=128, batch_size=128, verbose=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002) 
    
    last_epoch = 0
    
    if load_model_path is not None:
        load = torch.load(load_model_path)
        model.load_state_dict(load['model'])
        optimizer.load_state_dict(load['optimizer'])
        last_epoch = load['epoch']
        loss = load['loss']
    
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, train_data in enumerate(dataloader):
            x = train_data['image'].to(device)
            x = x.float().to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x) 
            loss.backward() 
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss = train_loss / len(dataloader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(last_epoch + epoch + 1, train_loss))
        
        if save_model_path is not None and (last_epoch +epoch + 1) % 10 == 0: 
            torch.save({
                'epoch': last_epoch + epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss
            }, save_model_path + "/epoch_{}.pt".format(last_epoch + epoch + 1))
        writer.add_scalar('training loss', train_loss, last_epoch + epoch)

            
#train settings
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
model = AutoEncoderSeq(color_mode="rgb", directory=None, loss_fn="L2", latent_space_dim=128, batch_size=128, verbose=True).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002) 
writer = SummaryWriter('runs/AE_L2')
#load_model_path = "D:/model/model_4/epoch_200.pt"
save_model_path = "D:/model/model_4"


#train
train(load_model_path=None, save_model_path=save_model_path, dataloader=train_hazelnut_dataloader, num_epochs=200, criterion=criterion, optimizer=optimizer, writer=writer)     
 
    
#visualize output
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
