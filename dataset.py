import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms

CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
              'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
            'pill', 'screw', 'toothbrush','transistor', 'zipper']

TEXTILES = ['carpet', 'grid', 'leather', 'tile', 'wood']


class MVTecADDataset(Dataset):
    def __init__(self, dataset_path, mode, category, color_mode, transform=None):
        """
        dataset_path (string): directory to MVTec AD dataset
        mode (string): create dataset for train, validation or test
        category(string): create dataset of specific category
        color_mode(string): create dataset 'grayscale' or 'rgb'
        transform(callable, optional): optional transform to be applied on a sample
        """
        assert category in CATEGORIES, "category: {} should be in {}".format(category, CATEGORIES)
        self.dataset_path = dataset_path
        self.mode = mode
        self.category = category
        self.color_mode = color_mode
        self.transform = transform
        
        self.transform_gray = transforms.Compose([
            transforms.Grayscale(1)
        ])

        self.image_dir, self.label, self.mask_dir = self.load_data_dir()
        self.image, self.mask = [], []

        for i, l, m in tqdm(zip(self.image_dir, self.label, self.mask_dir),
                            "| Loading augmented images | {} | {} |".format(self.mode, self.category)):

            image = Image.open(i).convert('RGB')
            if l == 0:
                mask = np.zeros((image.size[0], image.size[1]))
            else:
                mask = Image.open(m)

            sample = {'image': np.array(image), 'label': l, 'mask': np.array(mask)}
            if self.transform:
                sample = self.transform(sample)

            self.image.append(sample['image'])
            self.mask.append(sample['mask'])
            
        if color_mode == 'grayscale':
            for i, img in tqdm(enumerate(self.image), '| Changing RGB images to grayscale |'):
                img = self.transform_gray(img)
                self.image[i] = img

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image, label, mask = self.image[idx], self.label[idx], self.mask[idx]
        return image, label, mask

   


class Resize(object):
    """
    Rescales images to the given size
    """
    def __init__(self, size):
        self.size = size
    
    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        height, width = image.shape[:2]
        new_height, new_width = self.size, self.size

        image = transform.resize(image, (new_height, new_width))
        mask = transform.resize(mask, (new_height, new_width))
        
        return {'image': image, 'label': label, 'mask': mask}
    
class RandomCrop(object):
    """
    Randomly crops the image
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        height, width = image.shape[:2]
        new_height, new_width = self.size, self.size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        image = image[top:top + new_height, left:left + new_width]
        mask = mask[top:top + new_height, left:left + new_width]

        return {'image': image, 'label': label, 'mask': mask}
    
class RandomTranslation(object):
    """
    Randomly translates the image
    """
    def __init__(self, max_amount):
        self.max_amount = max_amount
    
    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        hshift, vshift = np.random.randint(0, self.max_amount), np.random.randint(0, self.max_amount)
        
        h, w, n = image.shape
        M = np.float32([[1,0,hshift],[0,1,vshift]])
  
        image = cv2.warpAffine(image, M, (h, w))
        mask = cv2.warpAffine(mask, M, (h, w)) 
        
        return {'image': image, 'label': label, 'mask': mask}
    
class RandomRotation(object):
    """
    Randomly rotataes the image
    """
    def __init__(self, max_amount):
        self.max_amount = max_amount
    
    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        angle = np.random.randint(0, self.max_amount) 
        
        image = transform.rotate(image, angle)
        mask = transform.rotate(mask, angle)     
        
        return {'image': image, 'label': label, 'mask': mask}
    
class ToTensor(object):
    """
    Converts ndarrays in sample to Tensors
    """
    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        if len(image.shape) == 3: image = image.transpose((2, 0, 1))
        if len(mask.shape) == 3: mask = mask.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': label, 'mask': torch.unsqueeze(torch.from_numpy(mask), 0)}
