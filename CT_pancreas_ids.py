"""
This code is for the 3D pancreas CT dataset reading
"""
import os
import sys
from monai import transforms
from monai.data.dataset import PersistentDataset, CacheDataset
import numpy as np
import random

import torch
from torch.utils.data import Dataset

class CachePanDataset(CacheDataset):
    def __init__(self, root, depth_size,
                 num_samples:int=12, ids:list=None,
                 cache_num: int = sys.maxsize, cache_rate: float = 1):
        self.root = root
        self.depth_size = depth_size
        self.num_samples = num_samples
        self.image_crop = 512
        # self.image_crop = 320
        self.keys = ('image', 'label')
        self.low_clip = -96
        self.high_clip = 215
        self.mean = 77.99
        self.std = 75.4

        self.transform = self.get_transform()
        self.data = self.get_data(ids=ids)
        super().__init__(self.data, self.transform, cache_num, cache_rate, num_workers=8)

    def get_transform(self):
        transform = transforms.Compose([
                transforms.LoadImaged(keys=self.keys),
                transforms.EnsureChannelFirstd(keys=self.keys),
                transforms.ScaleIntensityRanged(keys=self.keys[0],
                                                a_min=self.low_clip,
                                                a_max=self.high_clip,
                                                b_min=(self.low_clip-self.mean)/self.std,
                                                b_max=(self.high_clip-self.mean)/self.std,
                                                clip=True),
                transforms.Spacingd(self.keys, pixdim=(0.5, 0.5, 2.), mode = ("bilinear", "nearest")),
                transforms.Orientationd(self.keys, axcodes = 'RAS'),
                transforms.RandCropByPosNegLabeld(keys = self.keys,
                                                  label_key=self.keys[1],
                                                  spatial_size = (self.image_crop, self.image_crop, self.depth_size),
                                                  pos = 0.7,
                                                  neg = 0.3),
                transforms.RandFlipd(self.keys, prob = 0.5, spatial_axis=0),
                transforms.RandRotate90d(self.keys, prob=0.5, spatial_axes=(0,1)),
                transforms.ToTensord(self.keys)
            ])
        return transform

    def get_data(self, ids):
        full_img_path = sorted(os.listdir(os.path.join(self.root, 'train_images')))
        full_label_path = sorted(os.listdir(os.path.join(self.root, 'train_masks')))
        self.img_path = [full_img_path[id] for id in ids]
        self.label_path = [full_label_path[id] for id in ids]
        data = [{'image': os.path.join(self.root, 'train_images', image_path),
                 'label': os.path.join(self.root, 'train_masks', label_path),}
                 for image_path, label_path in zip(self.img_path, self.label_path)]
        return data

class PanCTDataset(Dataset):
    def __init__(self, root, depth_size, 
                 num_samples:int=12, is_transform:bool=True):
        super().__init__()
        self.root = root
        self.is_transform = is_transform
        self.depth_size = depth_size
        self.num_samples = num_samples
        self.image_crop = 512
        self.full_img_path = sorted(os.listdir(os.path.join(self.root, 'train_images')))
        self.full_label_path = sorted(os.listdir(os.path.join(self.root, 'train_masks')))
        
        self.keys = ('image', 'label')
        # self.low_clip = -150
        self.low_clip = -91
        self.high_clip = 250
        self.mean = 86.9
        self.std = 39.4

        self.prob = 0.4
        self.positive = 0.8

        self.transform = transforms.Compose([
                transforms.EnsureChannelFirstd(keys=self.keys),
                transforms.RandCropByPosNegLabeld(keys=self.keys,
                                                  label_key=self.keys[1],
                                                  spatial_size = (self.image_crop, 
                                                                  self.image_crop,
                                                                  self.depth_size,),
                                                  pos = 0.7,
                                                  neg = 0.3,
                                                  num_samples=self.num_samples),
                transforms.RandRotated(keys=self.keys, 
                                       range_x=np.pi/9,
                                       range_y=np.pi/9, 
                                       range_z=np.pi/9,
                                       mode=('bilinear', 'bilinear'),
                                       align_corners=True),
                transforms.RandAdjustContrastd(keys='image', prob = self.prob),
                transforms.RandZoomd(keys=self.keys, prob=self.prob,
                                     min_zoom=0.7, max_zoom=1.3,
                                     mode=('trilinear', 'trilinear'),
                                     align_corners=True),
                transforms.RandFlipd(keys=self.keys, prob=self.prob, spatial_axis=(0, 1)),
                transforms.ToTensord(keys=self.keys),
            ])

    def __len__(self) -> int:
        return len(self.full_img_path)
    
    def __str__(self) -> str:
        return "CT pancreas dataset"

    def __getitem__(self, index):
        temp_img_path = self.full_img_path[index]
        temp_label_path = self.full_label_path[index]

        img = np.load(os.path.join(self.root, 'data', temp_img_path))
        label = np.load(os.path.join(self.root, 'label', temp_label_path))

        img[img < self.low_clip] = self.low_clip
        img[img > self.high_clip] = self.high_clip
        img = (img - self.mean) / self.std
        img = img.transpose((1, 2, 0))
        label = label.transpose((1, 2, 0))
        img = img.astype(np.float32)
        label = (label>0.5).astype(np.uint8)
        data_dict = {'image': img,
                     'label': label,}
        data_dict = self.transform(data_dict)

        img = torch.stack([data_dict[i]['image'] for i in range(self.num_samples)], dim=0)
        label = torch.stack([(data_dict[i]['label']>=0.5).to(torch.uint8) for i in range(self.num_samples)], dim=0)
        return img, label


class IdPosPanCTDataset(Dataset):
    def __init__(self, root, depth_size, 
                 num_samples:int=12, is_transform:bool=True, ids:list=None):
        super().__init__()
        self.root = root
        self.is_transform = is_transform
        self.depth_size = depth_size
        self.num_samples = num_samples
        self.image_crop = 512
        self.full_img_path = sorted(os.listdir(os.path.join(self.root, 'image')))
        self.full_label_path = sorted(os.listdir(os.path.join(self.root, 'label')))
        self.img_path = [self.full_img_path[id] for id in ids]
        self.label_path = [self.full_label_path[id] for id in ids]
        self.keys = ('image', 'label')
        # self.low_clip = -150
        self.low_clip = -91
        self.high_clip = 250
        self.mean = 86.9
        self.std = 39.4

        self.prob = 0.4
        self.positive = 0.8

        self.transform = transforms.Compose([
                transforms.AddChanneld(keys=self.keys),
                transforms.RandCropByPosNegLabeld(keys=self.keys,
                                                  label_key=self.keys[1],
                                                  spatial_size = (self.image_crop, 
                                                                  self.image_crop,
                                                                  self.depth_size,),
                                                  pos = 0.7,
                                                  neg = 0.3,
                                                  num_samples=self.num_samples),
                transforms.RandRotated(keys=self.keys, 
                                       range_x=np.pi/9,
                                       range_y=np.pi/9, 
                                       range_z=np.pi/9,
                                       mode=('bilinear', 'bilinear'),
                                       align_corners=True),
                transforms.RandAdjustContrastd(keys='image', prob = self.prob),
                transforms.RandZoomd(keys=self.keys, prob=self.prob,
                                     min_zoom=0.7, max_zoom=1.3,
                                     mode=('trilinear', 'trilinear'),
                                     align_corners=True),
                transforms.RandFlipd(keys=self.keys, prob=self.prob, spatial_axis=(0, 1)),
                transforms.ToTensord(keys=self.keys),
            ])

    def __len__(self) -> int:
        return len(self.img_path)
    
    def __str__(self) -> str:
        return "CT pancreas dataset"

    def __getitem__(self, index):
        temp_img_path = self.img_path[index]
        temp_label_path = self.label_path[index]

        img = np.load(os.path.join(self.root, 'image', temp_img_path))
        label = np.load(os.path.join(self.root, 'label', temp_label_path))

        img[img < self.low_clip] = self.low_clip
        img[img > self.high_clip] = self.high_clip
        img = (img - self.mean) / self.std
        img = img.transpose((1, 2, 0))
        label = label.transpose((1, 2, 0))
        img = img.astype(np.float32)
        label = label.astype(np.uint8)
        print('org', label.dtype)
        '''
        print('path', temp_img_path)
        print('dtype', label.dtype)
        print('max value', np.max(label))
        print('shape', label.shape)
        '''
        # label = (label>0.5).astype(np.uint8)
        data_dict = {'image': img,
                     'label': label,}
        data_dict = self.transform(data_dict)

        img = torch.stack([data_dict[i]['image'] for i in range(self.num_samples)], dim=0)
        print('before', (data_dict[0]['label']).dtype)
        label = torch.stack([(data_dict[i]['label']).to(torch.uint8) for i in range(self.num_samples)], dim=0)
        print('after', label.dtype)
        return img, label


class EvaPanCTDataset(Dataset):
    def __init__(self, root, depth_size, ids:list=None):
        super().__init__()
        self.root = root
        self.depth_size = depth_size

        self.full_img_path = sorted(os.listdir(os.path.join(self.root, 'test_images')))
        self.full_label_path = sorted(os.listdir(os.path.join(self.root, 'test_masks')))
        self.img_path = [self.full_img_path[id] for id in ids]
        self.label_path = [self.full_label_path[id] for id in ids]
        self.keys = ('image', 'label')
        # self.low_clip = -150
        # for 1354 experiment
        '''
        self.low_clip = -96
        self.high_clip = 215
        '''
        self.low_clip = -91
        # self.high_clip = 183
        self.high_clip = 250
        self.mean = 86.9
        self.std = 39.4

        self.image_crop = 256

        self.transform = transforms.Compose([
            transforms.ToTensord(keys=self.keys),
            transforms.EnsureChannelFirstd(keys=self.keys),
            # transforms.Resized(keys=self.keys,
            #                       spatial_size=(-1, self.image_crop, self.image_crop)),
            # transforms.ScaleIntensityd(keys='image', minv=0, maxv=1),
            ])

    def __len__(self) -> int:
        return len(self.img_path)
    
    def __str__(self) -> str:
        return "MRI pancreas dataset"

    def __getitem__(self, index):
        temp_img_path = self.img_path[index]
        temp_label_path = self.label_path[index]

        img = np.load(os.path.join(self.root, 'test_images', temp_img_path))
        label = np.load(os.path.join(self.root, 'test_masks', temp_label_path))

        img[img < self.low_clip] = self.low_clip
        img[img > self.high_clip] = self.high_clip
        img = (img - self.mean) / self.std
        img = img.astype(np.float32)
        label = (label>0.5).astype(np.uint8)
        '''
        pos_index = np.sum(label, axis=(1, 2), keepdims=False)>0
        index = np.where(pos_index)
        min_index = np.min(index)
        max_index = np.max(index)
        if (max_index - min_index) < self.depth_size:
            center = min_index + max_index
            min_index = center - self.depth_size//2
            max_index = center + self.depth_size//2
            if min_index < 0:
                min_index = 0
                max_index = self.depth_size
            if max_index >= img.shape[0]:
                min_index = img.shape[0]-self.depth_size
                max_index = img.shape[0]
        # min_index = max(min_index//2, min_index-self.depth_size)
        img = img[min_index:max_index]
        label = label[min_index:max_index]
        '''
        data_dict = {'image': img,
                     'label': label}
        
        data_dict = self.transform(data_dict)

        img, label = data_dict['image'].permute(0, 2, 3, 1), data_dict['label'].permute(0, 2, 3, 1)
        return img, label
