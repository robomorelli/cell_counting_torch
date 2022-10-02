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


import argparse
import shutil

from augmentation_utils import *
from utils import *
import sys
sys.path.append('../config.py')
from config import *

def main(args):

    if args.color == 'r':
        if args.self_supervised:
            label_dict = {}
            CropImages = SelfSuperImagesR
            CropWeightedMasks = SelfSuperMasksR #the are not weighted but in this way we do not rename
            AugCropImagesBasic = AugSelfSuperImagesR
            AugCropMasksBasic = AugSelfSuperMasksR
            #the other path are not required for self_supervised
        else:
            CropImages = CropImagesR
            CropWeightedMasks =  CropWeightedMasksR
            AugCropImages = AugCropImagesR
            AugCropMasks = AugCropMasksR
            AugCropImagesBasic = AugCropImagesBasicR
            AugCropMasksBasic = AugCropMasksBasicR
            AugCropImagesAE = AugCropImagesAER
            AugCropMasksAE = AugCropMasksAER

    image_ids = os.listdir(CropImages)
    image_ids.sort()

    if args.color == 'y':
        CropImages = CropImages
        CropWeightedMasks =  CropWeightedMasks
        AugCropImages = AugCropImages
        AugCropMasks = AugCropMasks
        AugCropImagesBasic = AugCropImagesBasic
        AugCropMasksBasic = AugCropMasksBasic
        AugCropImagesAE = AugCropImagesAE
        AugCropMasksAE = AugCropMasksAE

        shift = len(image_ids)
        id_edges = [300, 302, 1161, 1380, 1908, 2064]  # These numbers are valid if use our test
        try:
            with open('../id_new_images.pickle', 'rb') as handle:
                dic = pickle.load(handle)
            id_new_images = dic['id_new_images']
        except:
            id_new_images = int(0.8 * shift)
        print(id_new_images)

    if args.start_from_zero:
        print('deleting existing files in destination folder')
        if not args.ae: #not autoencoder
            if (args.no_artifact_aug) | (args.unique_split):
                try:
                    shutil.rmtree(AugCropImagesBasic)
                except:
                    pass
                os.makedirs(AugCropImagesBasic, exist_ok=True)
                try:
                    shutil.rmtree(AugCropMasksBasic)
                except:
                    pass
                os.makedirs(AugCropMasksBasic, exist_ok=True)
                path_images = AugCropImagesBasic
                path_masks = AugCropMasksBasic
            else:
                try:
                    shutil.rmtree(AugCropImages)
                except:
                    pass
                os.makedirs(AugCropImages, exist_ok=True)
                try:
                    shutil.rmtree(AugCropMasks)
                except:
                    pass
                os.makedirs(AugCropMasks, exist_ok=True)
                path_images = AugCropImages
                path_masks = AugCropMasks

        else:
            try:
                shutil.rmtree(AugCropImagesAE)
            except:
                pass
            os.makedirs(AugCropImagesAE, exist_ok=True)

            os.makedirs(AugCropMasksAE, exist_ok=True)
            path_images = AugCropImagesAE
            path_masks = AugCropMasksAE

    src_files = os.listdir(CropImages)
    if args.copy_images:
        print('copying images')
        if args.self_supervised:
            old_names = src_files
            for file_name in src_files:
                full_file_name = os.path.join(CropImages, file_name)
                if os.path.isfile(full_file_name):
                    label = file_name.split('label')[-2]
                    new_name = file_name.split('_label')[0] + '.tiff'
                    label_dict[new_name] = label
                    shutil.copy(full_file_name, os.path.join(path_images, new_name))
            image_ids = os.listdir(path_images)
            image_ids.sort()
        else:
            for file_name in src_files:
                full_file_name = os.path.join(CropImages, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, path_images)

        if args.ae:
            print('copying images in masks for ae')
            for file_name in src_files:
                full_file_name = os.path.join(CropImages, file_name)
                if os.path.isfile(full_file_name):
                    new_name = file_name.split('_label')[0] + '.tiff'
                    shutil.copy(full_file_name, path_masks)

    if args.copy_masks and not args.ae:
        print('copying masks')
        if args.self_supervised:
            for file_name in src_files:
                full_file_name = os.path.join(CropWeightedMasks, file_name)
                if os.path.isfile(full_file_name):
                    new_name = file_name.split('_label')[0] + '.tiff'
                    shutil.copy(full_file_name, shutil.copy(full_file_name, os.path.join(path_masks, new_name)))
        else:

            for file_name in src_files:
                full_file_name = os.path.join(CropWeightedMasks, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, path_masks)

    if args.color == 'y':
        print('TO IMPLEMENT')

    elif args.color == 'r':
        SaveAugImages = path_images #saved in the same path where the starting one are copied
        SaveAugMasks = path_masks#saved in the same path where the starting one are copied
        if args.ae:
            print('autoencoder')
            for ix, name in enumerate(image_ids):

                image, mask = read_image_masks(name, path_images, path_masks)
                split_num = args.unique_split

                for i in range(split_num):
                    new_image = data_aug_red_ae(image, name)
                    new_mask = mask
                    aug_img_dir = SaveAugImages + '{}_{}.tiff'.format(name.split('.')[0], i)
                    aug_mask_dir = SaveAugMasks + '{}_{}.tiff'.format(name.split('.')[0], i)

                    plt.imsave(fname=aug_img_dir, arr=new_image)
                    plt.imsave(fname=aug_mask_dir, arr=new_mask)
                print(name.split('.')[0])

        else:

            for ix, name in enumerate(image_ids):
                label = old_names[ix].split('label')[-2]
                image, mask = read_image_masks(name, path_images, path_masks)
                minimum = mask[:, :, 1:2].min()
                maximum = mask[:, :, 1:2].max()
                labels_tar, nlabels_tar = ndimage.label(np.squeeze(mask[:, :, 0:1]))
                split_num = args.unique_split

                for i in range(split_num):
                    if not args.self_supervised:
                        new_image, new_mask = data_aug_red(image, mask, name, nlabels_tar, minimum, maximum)
                        aug_img_dir = SaveAugImages + '{}_{}.tiff'.format(name.split('.')[0], i)
                        aug_mask_dir = SaveAugMasks + '{}_{}.tiff'.format(name.split('.')[0], i)
                        plt.imsave(fname=aug_img_dir, arr=new_image)
                        plt.imsave(fname=aug_mask_dir, arr=new_mask)
                    else:
                        new_image, new_mask = data_aug_redSS(image, mask, name, nlabels_tar, minimum, maximum)
                        #new_name = name.split('label')[0]
                        label_dict['{}_{}.tiff'.format(name.split('.')[0], i)] = label
                        aug_img_dir = SaveAugImages + '{}_{}.tiff'.format(name.split('.')[0], i)
                        aug_mask_dir = SaveAugMasks + '{}_{}.tiff'.format(name.split('.')[0], i)
                        plt.imsave(fname=aug_img_dir, arr=new_image)
                        plt.imsave(fname=aug_mask_dir, arr=new_mask)

                print(name.split('.')[0])
            with open('images_labels.csv', 'w') as f:
                for key in label_dict.keys():
                    f.write("%s, %s\n" % (key, label_dict[key]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Define augmentation setting....default mode to follow the paper description')
    parser.add_argument('--start_from_zero', action='store_false',
                        help='remove previous file in the destination folder')
    parser.add_argument('--split_num', nargs="?", type=int, default=4,
                        help='augmentation factor for the images segmented automatically')
    parser.add_argument('--split_num_new_images', nargs="?", type=int, default=10,
                        help='augmentation factor fot the images segmented manually')
    parser.add_argument('--copy_images', default=True,
                        help='copy cropped in crop_aug images')
    parser.add_argument('--copy_masks', default=True,
                        help='copy cropped in crop_aug masks')
    parser.add_argument('--unique_split', type=int, default=2,
                        help='default is 0, define a different number and the same split factor will be used for all the images '
                             'for yellow we let 0 an pply different split among different images (automatically and manually segmented)'
                             'if the number is 0:the split_num and split_num_new_images num will be applied for the autotamitaccli'
                             ' and manually segmentede images respectively')
    parser.add_argument('--no_artifact_aug', action='store_const', const=True, default=True, # dafault to switch on False again to make ae aug
                        help='run basic augmentation')
    parser.add_argument('--color', nargs="?", type=str, default='r', help='color specification (y or r) to save pickle with right suffix'
                                                                         'it is needed only when maximum=False and max value among'
                                                                         'weights need to be found')
    parser.add_argument('--ae', action='store_const', const=True, default=False,
                        help='autoencoder')
    parser.add_argument('--self_supervised', action='store_const', const=True, default=False,
                        help='autoencoder')

    args = parser.parse_args()
    main(args)
