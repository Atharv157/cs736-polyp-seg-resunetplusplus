import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import cv2
import torch
import random
import glob
import matplotlib.pyplot as plt

from skimage import io
from utils.polyps_augmentation import *


class PolypsDataset(Dataset):
    def __init__(self, imgs_dir, labels_dir, transform=None):
        # self.imgs = sorted(glob.glob(os.path.join(imgs_dir, '*.jpg')))
        # self.masks = sorted(glob.glob(os.path.join(labels_dir, '*.jpg')))
        self.imgs = sorted(glob.glob(os.path.join(imgs_dir, '*.tif')))
        self.masks = sorted(glob.glob(os.path.join(labels_dir, '*.tif')))
        self.transform = transform

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):

        # for resunet and resunte++
        image = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        # image = cv2.imread(self.imgs[idx], cv2.IMREAD_GRAYSCALE) / 255.
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE) / 255.
        
        # for resunet and resunte++
        sample = {
            'image': image,
            'mask': mask[:, :, np.newaxis],
        }
        # sample = {
        #     'image': image[:, :, np.newaxis],
        #     'mask': mask[:, :, np.newaxis],
        # }
        


        if self.transform is not None:
            sample = self.transform(sample)

        # print(f"POLYPS DATASET : IMAGE : {sample['image'].shape}  ,   MASK : {sample['mask'].shape}")
        return sample
        

class JointTrTransform:
    def __init__(self, train=True):
        # define the *geometric* transforms you want to apply to both
        if train:
            self.transform = v2.Compose([
                # v2.ToImage(),
                v2.RandomResizedCrop(size=(256,256), scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.ToDtype(torch.float32, scale=True),
            ])

        self.image_only = v2.Compose([
            v2.ToImage(),
            v2.Normalize([0.5], [0.5])
        ])

    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]
        img = self.image_only(img)
        img = self.transform(img)
        mask = self.transform(mask)
        return {"image": img, "mask": mask}

class GrayscaleNormalization:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        image, mask = data['image'], data['mask']
        image = (image - self.mean) / self.std
        sample = {
            "image" : image,
            "mask" : mask
        }
        return sample
    

class ToTensor:
    def __call__(self, data):
        image, mask = data['image'], data['mask']

        # print(f"image : {image.shape} label : {mask.shape}")
        
        image = image.transpose(2, 0, 1).astype(np.float32)  # torch 의 경우 (C, H, W)
        mask = mask.transpose(2, 0, 1).astype(np.float32)
        
        sample = {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask),
        }
        return sample
    

class GrayscaleNormalizationSI:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        
    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image
    

class ToTensorSI:
    def __call__(self, image):

        image = image.transpose(2, 0, 1).astype(np.float32) 

        return image
    
class RandomFlip:
    def __call__(self, data):
        image, mask = data['image'], data['mask']
        
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            
        if np.random.rand() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
            
        ret = {
            'image': image,
            'mask': mask,
        }
        return ret

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, mask = data["image"], data["mask"]
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)
        sample = {
            "image" : image,
            "mask" : mask
        }
        # print("RESIZE")
        # print(sample['mask'].shape)
        return sample

class RandomCrop:
    def __init__(self, crop_size, size):
        self.crop_size = crop_size
        self.size = size

    def __call__(self, data):
        image, mask = data["image"], data["mask"]
        image, mask =  random_crop(image, mask, self.crop_size, self.size)
        sample = {
            "image" : image,
            "mask" : mask
        }
        return sample

class HorizontalFlip:
    def __init__(self, p=0.5, size=(256, 256)):
        self.p = p
        self.size = size

    def __call__(self, data):
        image, mask = data["image"], data["mask"]
        if random.random() < self.p:
            image, mask =  horizontal_flip(image, mask, self.size)
        sample = {
            "image" : image,
            "mask" : mask
        }
        return sample
    
class VerticalFlip:
    def __init__(self, p=0.5, size=(256, 256)):
        self.p = p
        self.size = size

    def __call__(self, data):
        image, mask = data["image"], data["mask"]
        if random.random() < self.p:
            image, mask =  vertical_flip(image, mask, self.size)
        sample = {
            "image" : image,
            "mask" : mask
        }
        return sample
    
class RandomRotation:
    def __init__(self, p=0.5, size=(256, 256)):
        self.p = p
        self.size = size

    def __call__(self, data):
        image, mask = data["image"], data["mask"]
        if random.random() < self.p:
            image, mask =  random_rotation(image, mask, self.size)
        sample = {
            "image" : image,
            "mask" : mask
        }
        return sample



def denormalization(data, mean, std):
    return (data * std) + mean

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy().transpose(1, 2, 0)  # (Batch, H, W, C)

# class BrightnessAugment:
#     def __init__(self, factor_range=(0.3, 0.9), size=(256, 256)):
#         self.factor_range = factor_range
#         self.size = size

#     def __call__(self, data):
#         image, mask = data["image"], data["mask"]
#         factor = random.uniform(*self.factor_range)
#         image, mask =  brightness_augment(image, mask, factor)
#         sample = {
#             "image" : image,
#             "mask" : mask
#         }
#         return sample    


class TestResize:
    def __init__(self, size):
        self.size = (size, size)  # Ensure (width, height) format

    def __call__(self, data):
        image, mask = data["image"], data["mask"]
        
        # Resize with proper interpolation
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        
        # Add channel dimension to mask if missing
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
            
        return {"image": image, "mask": mask}