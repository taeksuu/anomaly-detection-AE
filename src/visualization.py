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

transform_texture = [RandomCrop(256), ToTensor()]
transform_object = [RandomTranslation(40), RandomRotation(30), Rescale(256), ToTensor()]

train_grid = MVTecADDataset(directory, category="grid", mode="train", transform=transforms.Compose(transform_texture))
validate_grid = MVTecADDataset(directory, category="grid", mode="validate", transform=transforms.Compose(transform_texture))
test_grid = MVTecADDataset(directory, category="grid", mode="test", transform=transforms.Compose(transform_texture))

train_pill = MVTecADDataset(directory, category="pill", mode="train", transform=transforms.Compose(transform_object))
validate_pill = MVTecADDataset(directory, category="pill", mode="validate", transform=transforms.Compose(transform_object))
test_pill = MVTecADDataset(directory, category="pill", mode="test", transform=transforms.Compose([Rescale(256), ToTensor()]))


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
        if len(shape) == 3: plt.imshow(sample['image'].numpy().transpose((1, 2, 0)))
        else: plt.imshow(sample['image'].numpy(), cmap='gray')
        plt.axis('off')
        ax1.set_title("image")

        ax2 = fig.add_subplot(1, 2, 2)
        plt.imshow(sample['mask'], cmap='gray')
        plt.axis('off')
        ax2.set_title("mask")
        
        plt.suptitle("<{} - {}>".format(dataset.category, sample['label']), y=0.9)

for idx in range(11, 12):
    visualize_dataset(train_grid, idx)
    visualize_dataset(validate_grid, idx)
    visualize_dataset(test_grid, idx)
    
    visualize_dataset(train_pill, idx)
    visualize_dataset(validate_pill, idx)
    visualize_dataset(test_pill, idx)
        
        
def visualize_batch(sample_batched):
    """
    For a batch of samples, show image with its corresponding mask.
    If an image doesn't have a mask, the area for the mask is empty.
    """
    batch_size = len(sample_batched['image'])
    fig = plt.figure()
    
    for i in range(batch_size):
        ax1 = fig.add_subplot(2, batch_size, i + 1)
        if len(sample_batched['image'][i].shape) == 3: plt.imshow(sample_batched['image'][i].numpy().transpose((1, 2, 0)))
        else: plt.imshow(sample_batched['image'][i].numpy(), cmap='gray')
        plt.axis('off')
        ax1.set_title("image")

        ax2 = fig.add_subplot(2, batch_size, batch_size + i + 1)
        plt.imshow(sample_batched['mask'][i], cmap='gray')
        plt.axis('off')
        ax2.set_title("mask")


train_grid_dataloader = DataLoader(train_grid, batch_size=4, shuffle=True, num_workers=0)
validate_grid_dataloader = DataLoader(validate_grid, batch_size=4, shuffle=True, num_workers=0)
test_grid_dataloader = DataLoader(test_grid, batch_size=4, shuffle=True, num_workers=0)

train_pill_dataloader = DataLoader(train_pill, batch_size=4, shuffle=True, num_workers=0)
validate_pill_dataloader = DataLoader(validate_pill, batch_size=4, shuffle=True, num_workers=0)
test_pill_dataloader = DataLoader(test_pill, batch_size=4, shuffle=True, num_workers=0)


for i_batch, sample_batched in enumerate(train_grid_dataloader):
    if i_batch == 0:
        visualize_batch(sample_batched)
        break

for i_batch, sample_batched in enumerate(test_pill_dataloader):
    if i_batch == 0:
        visualize_batch(sample_batched)
        break
