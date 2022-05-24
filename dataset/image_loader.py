import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
import os
import cv2
import random
from torchvision.transforms import Normalize

class CellsLoader(Dataset):
    def __init__(self, images_path, masks_path, val_split=0.3, transform=None):
        self.imgs_path = images_path
        self.masks_path = masks_path
        self.val_split = val_split
        self.transform = transform

        self.file_list = os.listdir(self.imgs_path)

    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_path + self.file_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_path + self.file_list[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            #print('normalization')
            img = self.transform(img)
            mask = self.transform(mask)
            #img = img*(1./255)
            #mask = mask*(1./255)

        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)
        return img_tensor, mask_tensor

