import torch
from torchvision.transforms import transforms as T
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class CellsLoader(Dataset):
    def __init__(self, images_path, masks_path, val_split=0.3, grayscale = False,
                 transform=None, ae=None, test=False, priority_list=[]):
        self.imgs_path = images_path
        self.masks_path = masks_path
        self.val_split = val_split
        self.transform = transform
        self.ae = ae
        self.test = test
        self.priority_list = priority_list
        self.grayscale = grayscale

        #if self.grayscale:
        #    self.transform_gray = transform.transforms.append(T.Resize((1040,1400)))


        if len(priority_list)==0:
            self.img_list = os.listdir(self.imgs_path)
            self.mask_list = os.listdir(self.masks_path)
        else:
            self.img_list = priority_list
            self.mask_list = priority_list

    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        if self.ae == 'ae' and not self.test:
            shift = np.random.randint(0,1,1)[0]
            if '_' in self.img_list[idx]:
               name =  self.img_list[idx].split('_')[0]
               img = cv2.imread(self.imgs_path + name + '_{}.tiff'.format(shift))
               img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
               mask = cv2.imread(self.imgs_path + name + '.tiff')
               mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(self.imgs_path + self.img_list[idx].split('.')[0] + '_{}.tiff'.format(shift))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(self.imgs_path + self.img_list[idx])
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                mask = self.transform(mask)
                if self.grayscale:
                    img = self.transform_gray(img)
                else:
                    img = self.transform(img)
            return img.float(), mask.float()
        else:
            img = cv2.imread(self.imgs_path + self.img_list[idx])
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_path + self.mask_list[idx])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                mask = self.transform(mask)
                img = self.transform(img)
            return img.float(), mask.float()

