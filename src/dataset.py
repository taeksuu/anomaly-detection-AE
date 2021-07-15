import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

plt.ion()


category = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut','leather', 'metal_nut', 
            'pill', 'screw', 'tile', 'toothbrush','transistor', 'wood', 'zipper']

objects = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
            'pill', 'screw', 'toothbrush','transistor', 'zipper']

textiles = ['carpet', 'grid', 'leather', 'tile', 'wood']


class MVTecADDataset(Dataset):
    
    def __init__(self, directory, mode, transform=None):
        """
        directory (string): directory to all images
        mode (string): select between train, validation or test
        transform(callable, optional): optional transform to be applied on a sample
        """
        #super(MVTecADDataset, self).__init__()
        
        self.directory = directory
        self.mode = mode
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
        
        def sample_images(folder):
            """
            Appends all images and masks in given dataset to self.samples
            """
            images_directory = filter(lambda f: f.is_file() and f.name.endswith(".png"), os.scandir(folder.path))
            images_list = sorted(images_directory, key=lambda e: e.name)
            images = map(lambda f: (f.path, folder.name), images_list)

            if folder.name == "good":
                images = map(lambda f:(f[0], f[1], None), images)
                images = map(lambda f: {'image': f[0], 'label': f[1], 'mask': f[2]}, images)
            else:
                images = map(lambda f:(f[0], f[1], image2mask(f[0])), images)
                images = map(lambda f: {'image': f[0], 'label': f[1], 'mask': f[2]}, images)

            self.samples.extend(images)

            
        if mode == "train":
            for category in categories:
                category_directory = os.path.join(directory, category, mode)
                folders = sorted(filter(lambda f: f.is_dir(), os.scandir(category_directory)), key=lambda x: x.name)
                self.classes.update(map(lambda f: f.name, folders))
                for folder in folders:
                    sample_images(folder)
        else:
            for category in categories:
                category_directory = os.path.join(directory, category, "test")
                folders = sorted(filter(lambda f: f.is_dir(), os.scandir(category_directory)), key=lambda x: x.name)
                self.classes.update(map(lambda f: f.name, folders))
                for folder in folders:
                    sample_images(folder)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
#         image_arr = io.imread(sample['image'])
#         if sample['mask'] == None: mask = sample['mask']
#         else: mask = io.imread(sample['mask'])
            
#         sample['image'] = image_arr
        
#         if self.mode != "test" and self.transform:
#             sample_fetched = self.transform(sample_fetched)
            
        return sample

    
    
        
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
        M = np.float32([[1, 0, hshift],[0, 1, vshift]])
        
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
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': label, 'mask': torch.from_numpy(mask)}
