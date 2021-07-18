import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#directory to all the images
directory = "D:\mvtec_anomaly_detection"

train = MVTecADDataset(directory, category="bottle", mode="train", transform=transforms.Compose([
    ToTensor()
]))

validate = MVTecADDataset(directory, category="bottle", mode="validate", transform=transforms.Compose([
    ToTensor()
]))

test = MVTecADDataset(directory, category="bottle", mode="test", transform=transforms.Compose([
    ToTensor()
]))

def visualize_dataset(dataset, idx):
    """
    Shows image of given index in dataset with its corresponding mask.
    If an image doesn't have a mask, the area for the mask is all black.
    """
    if len(dataset) <= idx: print("out of index")
    else:
        sample = dataset[idx]
        shape = sample['image'].shape
        fig = plt.figure()

        ax1 = fig.add_subplot(1, 2, 1)
        plt.imshow(sample['image'])
        plt.axis('off')
        ax1.set_title("image")

        ax2 = fig.add_subplot(1, 2, 2)
        if sample['mask'] is None:            
            plt.imshow(np.zeros(shape))
        else:
            plt.imshow(sample['mask'], cmap='gray')
        plt.axis('off')
        ax2.set_title("mask")
        
        plt.suptitle("<{} - {}>".format(dataset.category, sample['label']), y=0.9)
        

for idx in range(3):
    visualize_dataset(train, idx)
    visualize_dataset(validate, idx)
    visualize_dataset(test, idx)
        
        
def visualize_batch(sample_batched):
"""
For a batch of samples, show image with its corresponding mask.
If an image doesn't have a mask, the area for the mask is empty.
"""
batch_size = len(sample_batched['image'])
fig = plt.figure()

for i in range(batch_size):
    ax1 = fig.add_subplot(2, batch_size, i + 1)
    plt.imshow(sample_batched['image'][i])
    plt.axis('off')
    ax1.set_title("image")

    ax2 = fig.add_subplot(2, batch_size, batch_size + i + 1)
    plt.imshow(sample_batched['mask'][i], cmap='gray')
    plt.axis('off')
    ax2.set_title("mask")


train_dataloader = DataLoader(train, batch_size=4, shuffle=True, num_workers=0)
validate_dataloader = DataLoader(validate, batch_size=4, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test, batch_size=4, shuffle=True, num_workers=0)


for i_batch, sample_batched in enumerate(train_dataloader):
    if i_batch == 0:
        visualize_batch(sample_batched)
        break
        
for i_batch, sample_batched in enumerate(validate_dataloader):
    if i_batch == 0:
        visualize_batch(sample_batched)
        break

for i_batch, sample_batched in enumerate(test_dataloader):
    if i_batch == 0:
        visualize_batch(sample_batched)
        break
