# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 Luca Clissa, Marco Dalla, Roberto Morelli
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Created on Tue May  7 10:42:13 2019
@author: Roberto Morelli
"""


import random
import numpy as np
import PIL

import sys
sys.path.append('../')
from config import *
from utils import *

from albumentations import (RandomCrop,CenterCrop,ElasticTransform,RGBShift,Rotate,
    Compose, ToFloat, FromFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,Transpose,
    ShiftScaleRotate, OpticalDistortion, GridDistortion, RandomBrightnessContrast, VerticalFlip, HorizontalFlip,

    HueSaturationValue,
)



IMG_WIDTH = 1600
IMG_HEIGTH = 1200

def lookup_tiff_aug(p = 0.5):

    return Compose([

        ToFloat(),
        #LOOKUP TABLE
        OneOf([
        RandomBrightnessContrast(brightness_limit=0,contrast_limit=(-0.7,0.0), p=0.7),
        HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.05, p=0.7),

            ], p=p),

        FromFloat(dtype='uint8', max_value=255.0),

            ], p=p)

def shifter_RGB(p = 0.5):

        return Compose([

        ToFloat(),
        #LOOKUP TABLE
        OneOf([
        RGBShift(r_shift_limit=[0.05,0.06], g_shift_limit=[0.04,0.045], b_shift_limit=0, p=1),
            ], p=p),

        FromFloat(dtype='uint8', max_value=255.0),

            ], p=p)

def shifter(p=.5):
    return Compose([
        ToFloat(),

        #ROTATION
        Rotate(limit=180, interpolation=1, border_mode=4, always_apply=False, p=0.75),
#         #FLIP
        OneOf([
            VerticalFlip(p = 0.6),
            HorizontalFlip(p = 0.6),
                ], p=p),

        FromFloat(dtype='uint8', max_value=255.0),

        ], p=p)

def elastic_def(alpha, alpha_affine, sigma, p=.5):
    return Compose([
        ToFloat(),

        ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, interpolation=1, border_mode=4,
                             always_apply=False, approximate=False,
                             p=1),
        ShiftScaleRotate(shift_limit=0.10, scale_limit=0, rotate_limit=(0, 0),
                         interpolation=1, border_mode=4, always_apply=False, p=0.3),

        FromFloat(dtype='uint8', max_value=255.0),


    ], p=p)

def edges_aug(p = 0.5):
    return Compose([

        ToFloat(),

        #LOOKUP TABLE
        OneOf([
        HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.10, val_shift_limit=0.1, p=0.75),
        RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.4,p=0.75),
        ], p=0.6),

        FromFloat(dtype='uint8', max_value=255.0),

    ], p=p)


def Gaussian(p=.5, blur_limit = 25):
    return Compose([
        ToFloat(),

            OneOf([
            Blur(blur_limit=25, p=1),
        ], p=1),

        FromFloat(dtype='uint8', max_value=255.0),


    ], p=p)



def data_aug(image ,mask, image_id, nlabels_tar, minimum, maximum):

    gaussian = random.random()
    generic_transf = random.random()
    elastic = random.random()
    resize = random.random()
    RGB = random.random()

    rows,cols,ch = image.shape
    rowsm,colsm,chm = mask.shape

    if (RGB < 0.05) & (nlabels_tar > 2):

        augmentation = shifter_RGB(p = 1)
        data = {"image": image}
        augmented = augmentation(**data)
        image = augmented["image"]

        augmentation = shifter(p = 0.5)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]

        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)

        # if gaussian <= 0.10:

            # gaussian_blur = Gaussian_y(p=1, blur_limit = 15)
            # data = {"image": image}
            # augmented = gaussian_blur(**data)
            # image = augmented["image"]

        return image, mask

    #65 before
    if generic_transf < 0.65:

        augmentation = lookup_tiff_aug(p = 0.7)
        data = {"image": image}
        augmented = augmentation(**data)
        image = augmented["image"]

        augmentation = shifter(p = 0.7)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]

        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)

        if gaussian <= 0.33:

            gaussian_blur = Gaussian(p=1, blur_limit = 15)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"]


        return image, mask


    if elastic < 0.9:

        alfa = random.choice([30, 30, 40, 40, 40 , 50, 60])
        alfa_affine = random.choice([40, 50, 50, 75, 75])
        sigma = random.choice([20, 30, 30, 40, 50])
        elastic = elastic_def(alfa, alfa_affine, sigma, p=1)
        data = {"image": image, "mask": mask}
        augmented = elastic(**data)
        image, mask = augmented["image"], augmented["mask"]

        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)

        return image, mask

    else:

        augmentation = shifter(p = 1)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]

        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)

        return image, mask

#     if resize <= 0.1:

#         res = 0.5
#         scaled_image = cv2.resize(image,(int(cols*res),int(rows*res))) # scale image if you want resize the input                                                                               andoutput image must be the same
#         scaled_mask = cv2.resize(mask,(int(cols*res),int(rows*res)))
#         bordersize = rows//4
#         b, g, r = cv2.split(image)
#         blu = b.mean()
#         green = g.mean()
#         red = r.mean()
#         image=cv2.copyMakeBorder(scaled_image, top=bordersize, bottom=bordersize, left=bordersize,
#                              right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[blu,green,red])
#         mask=cv2.copyMakeBorder(scaled_mask, top=bordersize, bottom=bordersize, left=bordersize,
#                             right=bordersize, borderType= cv2.BORDER_CONSTANT)

#         mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)

#         return image, mask

def data_aug_red(image, mask, image_id, nlabels_tar, minimum, maximum):

    gaussian = random.random()
    generic_transf = random.random()
    elastic = random.random()
    distorted = random.random()

    if generic_transf < 0.65:

        augmentation = lookup_tiff_augR(p=1)
        data = {"image": image}
        augmented = augmentation(**data)
        image = augmented["image"]

        augmentation = shifterR(p=0.8)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]

        mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        if gaussian <= 0.30:
            gaussian_blur = GaussianR(p=1, blur_limit=15)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"]

            mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        return image, mask

    if elastic < 0.95:

        alfa = random.choice([50, 60, 60, 65, 65, 65, 70])
        alfa_affine = random.choice([20, 20, 35, 40, 40])
        sigma = random.choice([10, 25, 15, 20, 20, 25])
        #         alfa = random.choice([350, 400])
        #         alfa_affine = random.choice([100, 150])
        #         sigma = random.choice([50, 75])
        elastic = elastic_defR(alfa, alfa_affine, sigma, p=1)
        data = {"image": image, "mask": mask}
        augmented = elastic(**data)
        image, mask = augmented["image"], augmented["mask"]

        mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        if gaussian <= 0.25:
            gaussian_blur = GaussianR(p=1, blur_limit=13)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"]

            mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        return image, mask

    if distorted < 1:

        augmentation = shifterR(p=1)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]

        if gaussian <= 0.15:
            gaussian_blur = GaussianR(p=1, blur_limit=15)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"]

        mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        return image, mask


def data_aug_redSS(image, mask, image_id, nlabels_tar, minimum, maximum):

    gaussian = random.random()
    generic_transf = random.random()
    elastic = random.random()
    distorted = random.random()

    if generic_transf < 0.65:

        augmentation = lookup_tiff_augR(p=1)
        data = {"image": image}
        augmented = augmentation(**data)
        image = augmented["image"]

        augmentation = shifterR(p=0.8)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]

        mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        if gaussian <= 0.80:
            gaussian_blur = GaussianR(p=1, blur_limit=31)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"]

            mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        return image, mask

    if elastic < 0.95:

        alfa = random.choice([50, 60, 60, 65, 65, 65, 70])
        alfa_affine = random.choice([20, 20, 35, 40, 40])
        sigma = random.choice([10, 25, 15, 20, 20, 25])
        #         alfa = random.choice([350, 400])
        #         alfa_affine = random.choice([100, 150])
        #         sigma = random.choice([50, 75])
        elastic = elastic_defR(alfa, alfa_affine, sigma, p=1)
        data = {"image": image, "mask": mask}
        augmented = elastic(**data)
        image, mask = augmented["image"], augmented["mask"]

        mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        if gaussian <= 0.25:
            gaussian_blur = GaussianR(p=1, blur_limit=13)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"]

            mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        return image, mask

    if distorted < 1:

        augmentation = shifterR(p=1)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]

        if gaussian <= 0.15:
            gaussian_blur = GaussianR(p=1, blur_limit=15)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"]

        mask[:, :, 1:2] = np.clip(mask[:, :, 1:2], minimum, maximum)

        return image, mask


def data_aug_red_ae(image, image_id):

    gaussian = random.random()
    generic_transf = random.random()
    #elastic = random.random()
    salt_pepper = random.random()

    if salt_pepper < 0.3:
        RGB_noise = SaltAndPepperNoise(noiseType="SnP")
        image = RGB_noise(image)

        return image

    if generic_transf < 0.65:
        augmentation = lookup_tiff_augR(p=1)
        data = {"image": image}
        augmented = augmentation(**data)
        image = augmented["image"]

        if gaussian <= 0.25:
            gaussian_blur = GaussianR(p=1, blur_limit=15)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"]

        return image

    else:
        gaussian_blur = GaussianR(p=1, blur_limit=15)
        data = {"image": image}
        augmented = gaussian_blur(**data)
        image = augmented["image"]

        return image




def lookup_tiff_augR(p=0.5):
    return Compose([

        ToFloat(),

        # LOOKUP TABLE
        OneOf([
            #         HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=1),
            RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.5, -0.3), p=0.2),
            #         HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=(-0.05,0.05), p=1),

        ], p=p),

        FromFloat(dtype='uint8', max_value=255.0),

    ], p=p)


def distortionR(p=0.5):
    return Compose([

        ToFloat(),

        # LOOKUP TABLE
        OneOf([

            GridDistortion(num_steps=4, distort_limit=0.5, interpolation=1, border_mode=cv2.BORDER_CONSTANT, p=1)

        ], p=p),

        FromFloat(dtype='uint8', max_value=255.0),

    ], p=p)


def shifterR(p=.5):
    return Compose([
        ToFloat(),

        # ROTATION
        Rotate(limit=180, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
               always_apply=False, p=0.75),
        #         #FLIP
        OneOf([
            VerticalFlip(p=0.6),
            HorizontalFlip(p=0.6),
        ], p=p),

        FromFloat(dtype='uint8', max_value=255.0),

    ], p=p)


def elastic_defR(alpha, alpha_affine, sigma, p=.5):
    return Compose([
        ToFloat(),

        ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, interpolation=1,
                         border_mode=cv2.BORDER_CONSTANT, always_apply=False,
                         approximate=False, p=1),

        FromFloat(dtype='uint8', max_value=255.0),

    ], p=p)


def GaussianR(p=.5, blur_limit=9):
    return Compose([
        ToFloat(),

        OneOf([
            Blur(blur_limit=blur_limit, p=1),
        ], p=1),

        FromFloat(dtype='uint8', max_value=255.0),

    ], p=p)


def data_aug_y(image_ids, images_path, masks_path, split_num, id_start_new_images,
                        split_num_new_images, id_edges, SaveAugImages, SaveAugMasks, ix, unique_split,
                        no_artifact_aug, ae=False):

    if no_artifact_aug:
        print('no artifact aug')

    if ae:
        tot = len(image_ids)
        for ax_index, image_id in enumerate(image_ids):

            ID = int(image_id.split('.')[0])
            image, _ = read_image_masks(image_id, images_path, masks_path)

            split_num_im = 1
            print('image {} on {} params: {}-{}'.format(ax_index, tot, ID, split_num_im))
            for i in range(split_num_im):
                elastic = random.random()
                RGB_noise = SaltAndPepperNoise(noiseType="SnP")
                new_image = RGB_noise(image)

                if elastic < 0.5:
                    alfa = random.choice([30, 30, 40, 40, 40, 50, 60])
                    alfa_affine = random.choice([40, 50, 50, 75, 75])
                    sigma = random.choice([20, 30, 30, 40, 50])
                    elastic = elastic_def(alfa, alfa_affine, sigma, p=1)
                    data = {"image": new_image}
                    augmented = elastic(**data)
                    new_image = augmented["image"]

                aug_img_dir = SaveAugImages + '{}_{}.tiff'.format(ID, i)
                ix += 1
                plt.imsave(fname=aug_img_dir, arr=new_image)
        return
    else:
        # for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
        tot = len(image_ids)
        for ax_index, image_id in enumerate(image_ids):
            ID_str = image_id.split('.')[0]
            ID = int(image_id.split('.')[0].split('_')[0])
            image, mask = read_image_masks(image_id, images_path, masks_path)
            minimum = mask[:, :, 1:2].min()
            maximum = mask[:, :, 1:2].max()
            labels_tar, nlabels_tar = ndimage.label(np.squeeze(mask[:, :, 0:1]))

            if unique_split == 0:
                if ID > id_start_new_images:
                    split_num_im = split_num_new_images
                else:
                    split_num_im = split_num
            else:
                split_num = unique_split

            print('image {} on {} params: {}-{}'.format(ax_index, tot, ID, split_num_im))

            if (ID_str in id_edges) & (not (no_artifact_aug)):
                print(ID, ix)
                for i in range(80):
                    image, mask = read_image_masks(image_id, images_path, masks_path)

                    augmentation = edges_aug(p=1)
                    data = {"image": image}
                    augmented = augmentation(**data)
                    new_image = augmented["image"]

                    augmentation = shifter(p=0.8)
                    data = {"image": new_image, "mask": mask}
                    augmented = augmentation(**data)
                    new_image, new_mask = augmented["image"], augmented["mask"]

                    new_mask[:, :, 1:2] = np.clip(new_mask[:, :, 1:2], minimum, maximum)

                    aug_img_dir = SaveAugImages + '{}_{}.tiff'.format(ID_str, i)
                    aug_mask_dir = SaveAugMasks + '{}_{}.tiff'.format(ID_str, i)
                    ix += 1

                    plt.imsave(fname=aug_img_dir, arr=new_image)
                    plt.imsave(fname=aug_mask_dir, arr=new_mask)

                for i in range(35):
                    image, mask = read_image_masks(image_id, images_path, masks_path)

                    alfa = random.choice([30, 30, 30, 40])
                    alfa_affine = random.choice([20, 20, 20, 30, 40, 40])
                    sigma = random.choice([20, 20, 20, 20, 30, 30, 15])

                    elastic = elastic_def(alfa, alfa_affine, sigma, p=1)
                    data = {"image": image, "mask": mask}
                    augmented = elastic(**data)
                    new_image, new_mask = augmented["image"], augmented["mask"]

                    new_mask[:, :, 1:2] = np.clip(new_mask[:, :, 1:2], minimum, maximum)

                    # '{}.tiff'.format(ix)
                    aug_img_dir = SaveAugImages + '{}_{}.tiff'.format(ID_str, i)
                    aug_mask_dir = SaveAugMasks + '{}_{}.tiff'.format(ID_str, i)
                    ix += 1

                    plt.imsave(fname=aug_img_dir, arr=new_image)
                    plt.imsave(fname=aug_mask_dir, arr=new_mask)

                for blur in range(1, 39, 3):
                    image, mask = read_image_masks(image_id, images_path, masks_path)

                    blur_limit = blur
                    gaussian_blur = Gaussian(p=1, blur_limit=blur_limit)
                    data = {"image": image}
                    augmented = gaussian_blur(**data)
                    new_image = augmented["image"]

                    aug_img_dir = SaveAugImages + '{}_{}.tiff'.format(ID_str, i)
                    aug_mask_dir = SaveAugMasks + '{}_{}.tiff'.format(ID_str, i)
                    ix += 1

                    plt.imsave(fname=aug_img_dir, arr=new_image)
                    plt.imsave(fname=aug_mask_dir, arr=mask)

            else:

                for i in range(split_num_im):
                    new_image, new_mask = data_aug(image, mask, image_id, nlabels_tar, minimum, maximum)

                    aug_img_dir = SaveAugImages + '{}_{}.tiff'.format(ID_str, i)
                    aug_mask_dir = SaveAugMasks + '{}_{}.tiff'.format(ID_str, i)
                    ix += 1

                    plt.imsave(fname=aug_img_dir, arr=new_image)
                    plt.imsave(fname=aug_mask_dir, arr=new_mask)

        return




#class RedAugumenter(object):
#    def __init__(self,
#                 treshold: float = 0.005,
#                 imgType: str = "cv2",
#                 lowerValue: int = 5,
#                 upperValue: int = 250,
#                 noiseType: str = "SnP"):
#        self.treshold = treshold
#        self.imgType = imgType
#        self.lowerValue = lowerValue  # 255 would be too high
#        self.upperValue = upperValue  # 0 would be too low



class SaltAndPepperNoise(object):
    r""" Implements 'Salt-and-Pepper' noise
    Adding grain (salt and pepper) noise
    (https://en.wikipedia.org/wiki/Salt-and-pepper_noise)

    assumption: high values = white, low values = black

    Inputs:
            - threshold (float):
            - imgType (str): {"cv2","PIL"}
            - lowerValue (int): value for "pepper"
            - upperValue (int): value for "salt"
            - noiseType (str): {"SnP", "RGB"}
    Output:
            - image ({np.ndarray, PIL.Image}): image with
                                               noise added
    """

    def __init__(self,
                 treshold: float = 0.005,
                 imgType: str = "cv2",
                 lowerValue: int = 5,
                 upperValue: int = 250,
                 noiseType: str = "SnP"):
        self.treshold = treshold
        self.imgType = imgType
        self.lowerValue = lowerValue  # 255 would be too high
        self.upperValue = upperValue  # 0 would be too low
        if (noiseType != "RGB") and (noiseType != "SnP"):
            raise Exception("'noiseType' not of value {'SnP', 'RGB'}")
        else:
            self.noiseType = noiseType
        super(SaltAndPepperNoise).__init__()

    def __call__(self, img):
        if self.imgType == "PIL":
            img = np.array(img)
        if type(img) != np.ndarray:
            raise TypeError("Image is not of type 'np.ndarray'!")

        if self.noiseType == "SnP":
            random_matrix = np.random.rand(img.shape[0], img.shape[1])
            img[random_matrix >= (1 - self.treshold)] = self.upperValue
            img[random_matrix <= self.treshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = np.random.random(img.shape)
            img[random_matrix >= (1 - self.treshold)] = self.upperValue
            img[random_matrix <= self.treshold] = self.lowerValue

        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            # return as PIL image for torchvision transforms compliance
            return PIL.Image.fromarray(img)


