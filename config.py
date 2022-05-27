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
import os
from pathlib import Path

IMG_WIDTH = 1600
IMG_HEIGHT = 1200

root = os.getcwd()
root = Path(root).parent.as_posix()

OriginalImages = root + '/DATASET/original_images/images/'
OriginalMasks = root + '/DATASET/original_masks/masks/'

NewImages = root + '/DATASET/new_images/images/'
NewMasks = root + '/DATASET/new_masks/masks/'

NewTestImages = root + '/DATASET/new_test/images/'
NewTestsMasks = root + '/DATASET/new_test/masks/'

# Temporary folder for the union of original and new images, final dataset_loader that we are going to share is going to be in
# DATASET/train_val/{images, masks}_before_crop and DATASET/test/all_{images,masks}
AllImages = root + '/DATASET/all_images/images/'
AllMasks = root + '/DATASET/all_masks/masks/'

if not os.path.exists(AllImages):
    os.makedirs(AllImages)

if not os.path.exists(AllMasks):
    os.makedirs(AllMasks)

#I'd like to change in this way
TrainValImages = root + '/DATASET/train_val/full_size/all_images/images/'
TrainValMasks = root + '/DATASET/train_val/full_size/all_masks/masks/'

if not os.path.exists(TrainValImages):
    os.makedirs(TrainValImages)

if not os.path.exists(TrainValMasks):
    os.makedirs(TrainValMasks)

TestImages = root + '/DATASET/test/all_images/images/'
TestMasks = root + '/DATASET/test/all_masks/masks/'

if not os.path.exists(TestImages):
    os.makedirs(TestImages)

if not os.path.exists(TestMasks):
    os.makedirs(TestMasks)

#This should be only for train_val
CropImages = root + '/DATASET/train_val/cropped/images/'
CropMasks = root + '/DATASET/train_val/cropped/masks/'
CropWeightedMasks = root + '/DATASET/train_val/cropped/weighted_masks/'

if not os.path.exists(CropImages):
    os.makedirs(CropImages)

if not os.path.exists(CropMasks):
    os.makedirs(CropMasks)
if not os.path.exists(CropWeightedMasks):
    os.makedirs(CropWeightedMasks)

# Final folder where the images for train reside, already cropped and augmented and weighted
# Cropped and augmented images are going to be read from the same folder
AugCropImages = root + '/DATASET/train_val/crop_augmented/images/'
AugCropMasks = root + '/DATASET/train_val/crop_augmented/masks/'

AugCropImagesSplitted = root + '/DATASET/train_val/crop_augmented_splitted_images/images/'
AugCropMasksSplitted = root + '/DATASET/train_val/crop_augmented_splitted_masks/masks/'

AugCropImagesBasic = root + '/DATASET/train_val/crop_augmented_basic/images/'
AugCropMasksBasic = root + '/DATASET/train_val/crop_augmented_basic/masks/'

AugCropImagesBasicSplitted = root + '/DATASET/train_val/crop_augmented_basic_splitted_images/images/'
AugCropMasksBasicSplitted = root + '/DATASET/train_val/crop_augmented_basic_splitted_masks/masks/'

if not os.path.exists(AugCropImages):
    os.makedirs(AugCropImages)

if not os.path.exists(AugCropMasks):
    os.makedirs(AugCropMasks)

if not os.path.exists(AugCropImagesBasic):
    os.makedirs(AugCropImagesBasic)

if not os.path.exists(AugCropMasksBasic):
    os.makedirs(AugCropMasksBasic)

ModelResults = root + '/model_results_torch/'

ModelResultsRay = root + '/model_results_torch_ray/'

if not os.path.exists(ModelResults):
    os.makedirs(ModelResults)

if not os.path.exists(ModelResultsRay):
    os.makedirs(ModelResultsRay)
