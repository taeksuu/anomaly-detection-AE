import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut','leather', 'metal_nut', 
            'pill', 'screw', 'tile', 'toothbrush','transistor', 'wood', 'zipper']

class MVTecADDataset(Dataset):
    def __init__(self, dataset_path, mode, category, transform=False):    
        assert category in CATEGORIES, "category: {} should be in {}".format(category, CATEGORIES)
        self.dataset_path = dataset_path
        self.mode = mode
        self.category = category
        self.transform = transform
        
        self.transform_image = transforms.Compose([
#             transforms.Resize(256),
#             transforms.ToTensor()
            transforms.Resize(256, Image.ANTIALIAS),
            transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose([
#             transforms.Resize(256),
#             transforms.ToTensor()
            transforms.Resize(256, Image.NEAREST),
            transforms.ToTensor()
        ])
        
        self.image_dir, self.label, self.mask_dir = self.load_data_dir()
        self.image, self.mask = [], []
        
        for image_dir in self.image_dir:
            image = Image.open(image_dir).convert('RGB')
            if self.transform:
                image = self.transform_image(image)
            self.image.append(image)
            
        for label, mask_dir in zip(self.label, self.mask_dir):
            if label == 0:
                mask = torch.zeros([1, self.image[0].shape[1], self.image[0].shape[2]])
            else:
                mask = Image.open(mask_dir)
                if self.transform:
                    mask = self.transform_mask(mask)
            self.mask.append(mask)
        
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        image, label, mask = self.image[idx], self.label[idx], self.mask[idx]       
        return image, label, mask
    
    def load_data_dir(self):
        image, label, mask = [], [], []
        mode = self.mode
        if mode == "validate": mode = "train"
        
        img_dir = os.path.join(self.dataset_path, self.category, mode)
        gt_dir = os.path.join(self.dataset_path, self.category, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            image.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                label.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                label.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(image) == len(label), 'number of images and labels should be same'
                 
        if self.mode == "test": return list(image), list(label), list(mask)
        else:
            N = len(list(image))
            if self.mode == "train": return list(image)[:int(0.9 * N)], list(label)[:int(0.9 * N)], list(mask)[:int(0.9 * N)]
            else: return list(image)[int(0.9 * N):], list(label)[int(0.9 * N):], list(mask)[int(0.9 * N):]
