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
root = Path(root).as_posix()

if root.split('/')[-1] != "cell_counting_torch":
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

AllImagesR = root + '/DATASET/fine_tuning/red/all_images/images/'
AllMasksR = root + '/DATASET/fine_tuning/red/all_masks/masks/'

if not os.path.exists(AllImages):
    os.makedirs(AllImages)

if not os.path.exists(AllMasks):
    os.makedirs(AllMasks)

if not os.path.exists(AllImagesR):
    os.makedirs(AllImagesR)

if not os.path.exists(AllMasksR):
    os.makedirs(AllMasksR)

#I'd like to change in this way
TrainValImages = root + '/DATASET/train_val/full_size/all_images/images/'
TrainValMasks = root + '/DATASET/train_val/full_size/all_masks/masks/'

TrainValImagesR = root + '/DATASET/fine_tuning/red/train_val/full_size/all_images/images/'
TrainValMasksR = root + '/DATASET/fine_tuning/red/train_val/full_size/all_masks/masks/'

if not os.path.exists(TrainValImages):
    os.makedirs(TrainValImages)

if not os.path.exists(TrainValMasks):
    os.makedirs(TrainValMasks)

if not os.path.exists(TrainValImagesR):
    os.makedirs(TrainValImagesR)

if not os.path.exists(TrainValMasksR):
    os.makedirs(TrainValMasksR)

TestImages = root + '/DATASET/test/all_images/images/'
TestMasks = root + '/DATASET/test/all_masks/masks/'
TestImagesR = root + '/DATASET/fine_tuning/red/test/all_images/images/'
TestMasksR = root + '/DATASET/fine_tuning/red/test/all_masks/masks/'

if not os.path.exists(TestImages):
    os.makedirs(TestImages)

if not os.path.exists(TestMasks):
    os.makedirs(TestMasks)

if not os.path.exists(TestImagesR):
    os.makedirs(TestImagesR)

if not os.path.exists(TestMasksR):
    os.makedirs(TestMasksR)

#This should be only for train_val
CropImages = root + '/DATASET/train_val/cropped/images/'
CropMasks = root + '/DATASET/train_val/cropped/masks/'
CropMasksSS = root + '/DATASET/train_val/cropped/ss_masks/'
CropWeightedMasks = root + '/DATASET/train_val/cropped/weighted_masks/'
CropWeightedMasksSS = root + '/DATASET/train_val/self_supervised/ss_weighted_masks/'

CropImagesR = root + '/DATASET/fine_tuning/red/train_val/cropped/images/'
CropMasksR = root + '/DATASET/fine_tuning/red/train_val/cropped/masks/'
CropWeightedMasksR = root + '/DATASET/fine_tuning/red/train_val/cropped/weighted_masks/'

if not os.path.exists(CropImages):
    os.makedirs(CropImages)
if not os.path.exists(CropMasks):
    os.makedirs(CropMasks)
    if not os.path.exists(CropMasksSS):
        os.makedirs(CropMasksSS)
if not os.path.exists(CropWeightedMasks):
    os.makedirs(CropWeightedMasks)
if not os.path.exists(CropWeightedMasksSS):
    os.makedirs(CropWeightedMasksSS)

if not os.path.exists(CropImagesR):
    os.makedirs(CropImagesR)
if not os.path.exists(CropMasksR):
    os.makedirs(CropMasksR)
if not os.path.exists(CropWeightedMasksR):
    os.makedirs(CropWeightedMasksR)

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

AugCropImagesAE = root + '/DATASET/train_val/crop_augmented_AE/images/'
AugCropMasksAE = root + '/DATASET/train_val/crop_augmented_AE/masks/'

AugCropImagesFS = root + '/DATASET/train_val/few_shot/images/'
AugCropMasksFS = root + '/DATASET/train_val/few_shot/masks/'


#AugCropMasksSS = root + '/DATASET/train_val/self_supervised/masks/'
AugCropMasksSS = root + '/DATASET/train_val/self_supervised/weighted_masks_augmented/'

if not os.path.exists(AugCropImages):
    os.makedirs(AugCropImages)
if not os.path.exists(AugCropMasks):
    os.makedirs(AugCropMasks)

if not os.path.exists(AugCropImagesBasic):
    os.makedirs(AugCropImagesBasic)
if not os.path.exists(AugCropMasksBasic):
    os.makedirs(AugCropMasksBasic)

if not os.path.exists(AugCropImagesAE):
    os.makedirs(AugCropImagesAE)
if not os.path.exists(AugCropMasksFS):
    os.makedirs(AugCropMasksFS)

if not os.path.exists(AugCropMasksSS):
    os.makedirs(AugCropMasksSS)

AugCropImagesR = root + '/DATASET/fine_tuning/red/train_val/crop_augmented/images/'
AugCropMasksR = root + '/DATASET/fine_tuning/red/train_val/crop_augmented/masks/'

AugCropImagesSplittedR = root + '/DATASET/fine_tuning/red/train_val/crop_augmented_splitted_images/images/'
AugCropMasksSplittedR = root + '/DATASET/fine_tuning/red/train_val/crop_augmented_splitted_masks/masks/'

AugCropImagesBasicR = root + '/DATASET/fine_tuning/red/train_val/crop_augmented_basic/images/'
AugCropMasksBasicR = root + '/DATASET/fine_tuning/red/train_val/crop_augmented_basic/masks/'

AugCropImagesBasicSplittedR = root + '/DATASET/fine_tuning/red/train_val/crop_augmented_basic_splitted_images/images/'
AugCropMasksBasicSplittedR = root + '/DATASET/fine_tuning/red/train_val/crop_augmented_basic_splitted_masks/masks/'

AugCropImagesAER = root + '/DATASET/fine_tuning/red/train_val/crop_augmented_AE/images/'
AugCropMasksAER = root + '/DATASET/fine_tuning/red/train_val/crop_augmented_AE/masks/'

AugCropImagesFewShotR = root + '/DATASET/fine_tuning/red/train_val/few_shot/images/'
AugCropMasksFewShotR = root + '/DATASET/fine_tuning/red/train_val/few_shot/masks/'

SelfSuperImagesR = root + '/DATASET/self_supervised/images/'
SelfSuperMasksR = root + '/DATASET/self_supervised/masks/'
AugSelfSuperImagesR = root + '/DATASET/self_supervised/aug_images/'
AugSelfSuperMasksR= root + '/DATASET/self_supervised/aug_masks/'

if not os.path.exists(SelfSuperImagesR):
    os.makedirs(SelfSuperImagesR)
if not os.path.exists(SelfSuperMasksR):
    os.makedirs(SelfSuperMasksR)

if not os.path.exists(AugSelfSuperImagesR):
    os.makedirs(AugSelfSuperImagesR)
if not os.path.exists(AugSelfSuperMasksR):
    os.makedirs(AugSelfSuperMasksR)

if not os.path.exists(AugCropImagesFewShotR):
    os.makedirs(AugCropImagesFewShotR)
if not os.path.exists(AugCropMasksFewShotR):
    os.makedirs(AugCropMasksFewShotR)

if not os.path.exists(AugCropImagesR):
    os.makedirs(AugCropImagesR)
if not os.path.exists(AugCropMasksR):
    os.makedirs(AugCropMasksR)

if not os.path.exists(AugCropImagesBasicR):
    os.makedirs(AugCropImagesBasicR)
if not os.path.exists(AugCropMasksBasicR):
    os.makedirs(AugCropMasksBasicR)


if not os.path.exists(AugCropImagesAER):
    os.makedirs(AugCropImagesAER)
if not os.path.exists(AugCropMasksAER):
    os.makedirs(AugCropMasksAER)

ModelResults = root + '/model_results/'
ModelResultsRay = root + '/model_results_ray/'

if not os.path.exists(ModelResults):
    os.makedirs(ModelResults)
if not os.path.exists(ModelResultsRay):
    os.makedirs(ModelResultsRay)


