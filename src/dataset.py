import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()


categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut','leather', 'metal_nut', 
            'pill', 'screw', 'tile', 'toothbrush','transistor', 'wood', 'zipper']

objects = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
            'pill', 'screw', 'toothbrush','transistor', 'zipper']

textiles = ['carpet', 'grid', 'leather', 'tile', 'wood']


class MVTecADDataset(Dataset):
    
    def __init__(self, directory, mode, category, transform=None):
        """
        directory (string): directory to all images
        mode (string): select between train, validation or test
        category (string): select a category among categories
        transform(callable, optional): optional transform to be applied on a sample
        """
        #super(MVTecADDataset, self).__init__()
        
        self.directory = directory
        self.mode = mode
        self.category = category
        self.transform = transform
        
        self.samples = []
        self.classes = set()
        
        def image2mask(directory):
            """
            Returns the corresponding mask directory for the given image directory if it has one(mode="validation", mode="test" & abnormal). 
            Else, returns None(mode="train",  mode="test" & normal).

            directory (string): directory to specific image
            """
            if self.mode == "train" or (self.mode == "test" and directory.find("good") >= 0): #directory에 또 다른 good이 들어가면 문제가 생김
                return None
            else:
                main, sub = directory.split("test") #directory에 또 다른 test가 들어가면 문제가 생김
                details, extension = sub.split(".")
                return main + "ground_truth" + details + "_mask." + extension
        
        def sample_images(folder, mode):
            """
            Appends all images and masks in given dataset to self.samples
            """
            images_directory = filter(lambda f: f.is_file() and f.name.endswith(".png"), os.scandir(folder.path))
            images_list = sorted(images_directory, key=lambda e: e.name)
            images = map(lambda f: (f.path, folder.name), images_list)

            if mode == "train":
                if folder.name == "good":
                    images = map(lambda f:(f[0], f[1], None), images)
                    images = map(lambda f: {'image': f[0], 'label': f[1], 'mask': f[2]}, images)
                else:
                    images = map(lambda f:(f[0], f[1], image2mask(f[0])), images)
                    images = map(lambda f: {'image': f[0], 'label': f[1], 'mask': f[2]}, images)
            elif mode == "validate":
                if folder.name == "good":
                    images = map(lambda f:(f[0], f[1], None), list(images)[::2])
                    images = map(lambda f: {'image': f[0], 'label': f[1], 'mask': f[2]}, images)
                else:
                    images = map(lambda f:(f[0], f[1], image2mask(f[0])), list(images)[::2])
                    images = map(lambda f: {'image': f[0], 'label': f[1], 'mask': f[2]}, images)
            else:
                if folder.name == "good":
                    images = map(lambda f:(f[0], f[1], None), list(images)[1::2])
                    images = map(lambda f: {'image': f[0], 'label': f[1], 'mask': f[2]}, images)
                else:
                    images = map(lambda f:(f[0], f[1], image2mask(f[0])), list(images)[1::2])
                    images = map(lambda f: {'image': f[0], 'label': f[1], 'mask': f[2]}, images)
            self.samples.extend(images)

            
        if self.mode == "train":
#             for category in categories:
            category_directory = os.path.join(directory, category, mode)
            folders = sorted(filter(lambda f: f.is_dir(), os.scandir(category_directory)), key=lambda x: x.name)
            self.classes.update(map(lambda f: f.name, folders))
            for folder in folders:
                sample_images(folder, mode)
        elif self.mode == "validate":
#             for category in categories:
            category_directory = os.path.join(directory, category, "test")
            folders = sorted(filter(lambda f: f.is_dir(), os.scandir(category_directory)), key=lambda x: x.name)
            self.classes.update(map(lambda f: f.name, folders))
            for folder in folders:
                sample_images(folder, mode)
        #self.mode == "test"
        else:
#             for category in categories:
            category_directory = os.path.join(directory, category, mode)
            folders = sorted(filter(lambda f: f.is_dir(), os.scandir(category_directory)), key=lambda x: x.name)
            self.classes.update(map(lambda f: f.name, folders))
            for folder in folders:
                sample_images(folder, mode)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]   
        
        image = io.imread(sample['image'])
        label = sample['label']
        shape = image.shape
        zero_mask = np.zeros((shape[0], shape[1]))
        
        if sample['mask'] is None: mask = zero_mask
        else: mask = io.imread(sample['mask'])
            
        sample_fetched = {'image': image, 'label': label, 'mask': mask}

        if self.mode != "test" and self.transform:
            sample_fetched = self.transform(sample_fetched)
            
        return sample_fetched
  
        
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
        if sample['mask'].any() != None: mask = mask[top:top + new_height, left:left + new_width]

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
        
        image_aug = cv2.warpAffine(image, M, (h, w))
        if sample['mask'].any() != None: mask_aug = cv2.warpAffine(mask, M, (h, w))
        
        return {'image': image_aug, 'label': label, 'mask': mask_aug}
    
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
        if sample['mask'].any() != None: mask = transform.rotate(mask, angle)
        
        return {'image': image, 'label': label, 'mask': mask}
    
class ToTensor(object):
    """
    Converts ndarrays in sample to Tensors
    """
    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        #image = image.transpose((2, 0, 1))
        if sample['mask'] is not None: 
            #mask.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image), 'label': label, 'mask': torch.from_numpy(mask)}
        else: return {'image': torch.from_numpy(image), 'label': label, 'mask': mask}
