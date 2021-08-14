import torch
from torch.utils.data import Dataset

import os
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut','leather', 'metal_nut', 
            'pill', 'screw', 'tile', 'toothbrush','transistor', 'wood', 'zipper']

OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
            'pill', 'screw', 'toothbrush','transistor', 'zipper']

TEXTILES = ['carpet', 'grid', 'leather', 'tile', 'wood']

class MVTecADDataset(Dataset):
    def __init__(self, directory, category, mode, transform):
        self.directory = directory
        self.category = category
        self.mode = mode
        self.transform = transform

        self.image_dir, self.label, self.mask_dir = self.get_data()

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_dir[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label[idx]
        if label == 1:
            mask = cv2.imread(self.mask_dir[idx], cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, label, mask

    def get_data(self):
        image_dir, label, mask_dir = [], [], []

        if self.mode == 'train':
            train_dir = os.path.join(self.directory, self.category, 'train', 'good')
            for dir in os.listdir(train_dir):
                image_dir.append(os.path.join(train_dir, dir))
                label.append(0)
                mask_dir.append(None)
        elif self.mode == 'test':
            for path, dir, files in os.walk(os.path.join(self.directory, self.category, 'test')):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if os.path.split(path)[-1] != 'good' and ext == '.png':
                        image_dir.append(os.path.join(path, filename))
                        label.append(1)
            for path, dir, files in os.walk(os.path.join(self.directory, self.category, 'ground_truth')):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if ext == '.png':
                        mask_dir.append(os.path.join(path, filename))
            for path, dir, files in os.walk(os.path.join(self.directory, self.category, 'test')):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if os.path.split(path)[-1] == 'good' and ext == '.png':
                        image_dir.append(os.path.join(path, filename))
                        label.append(0)
                        mask_dir.append(None)

        return image_dir, label, mask_dir
