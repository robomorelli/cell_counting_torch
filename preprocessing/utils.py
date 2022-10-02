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

import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import erosion
from skimage.morphology import remove_small_holes, remove_small_objects,\
label, erosion, dilation, local_maxima, skeletonize, binary_erosion, remove_small_holes
from skimage.segmentation import watershed
from scipy import ndimage
import tqdm
import random
import pickle
#from keras import backend as K
#import tensorflow as tf
#
from config import *

def read_masks(path, image_id):

        mask = cv2.imread(path + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        return mask

def read_images(path, image_id):

        img = cv2.imread(path + image_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

def read_image_masks(image_id, images_path,  masks_path):

        image = cv2.imread(images_path + image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(masks_path + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        return image, mask

def read_image_masks_r(image_id, images_path, masks_path):
    x = cv2.imread(images_path + image_id)
    image = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    mask_id = image_id.split('.')[0] + '_mask.tiff'
    mask = cv2.imread(masks_path + mask_id)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    return image, mask

#def cropper(image, mask, NumCropped, XCropSize, YCropSize
    #            ,x_coord, y_coord ,XCropNum, YCropNum, XShift, YShift):

    #CroppedImgs = np.zeros((NumCropped, YCropSize, XCropSize, 3), np.uint8)
    #CroppedMasks = np.zeros((NumCropped, YCropSize, XCropSize), np.uint8)
    #idx = 0

    #for i in range(0, XCropNum):
    #    for j in range(0, YCropNum):

    #            if (i == 0) & (j == 0):
    #                CroppedImgs[idx] = image[y_coord[j]:y_coord[j+1] + YShift, x_coord[i]:x_coord[i+1] + XShift]
    #                CroppedMasks[idx] = mask[y_coord[j]:y_coord[j+1] + YShift, x_coord[i]:x_coord[i+1] + XShift]
    #                idx +=1

    #            if (i == 0) & (j != 0):
    #                CroppedImgs[idx] = image[y_coord[j] - YShift : y_coord[j+1], x_coord[i]:x_coord[i+1] + XShift]
    #                CroppedMasks[idx] = mask[y_coord[j] - YShift : y_coord[j+1], x_coord[i]:x_coord[i+1] + XShift]
    #                idx +=1

    #            if (i != 0) &  (j == 0):
    #                CroppedImgs[idx] = image[y_coord[j]:y_coord[j+1] + YShift, x_coord[i] - XShift :x_coord[i+1]]
    #                CroppedMasks[idx] = mask[y_coord[j]:y_coord[j+1] + YShift, x_coord[i] - XShift :x_coord[i+1]]
    #                idx +=1

    #            if (i != 0) &  (j != 0):
    #                CroppedImgs[idx] = image[y_coord[j] - YShift : y_coord[j+1], x_coord[i] - XShift :x_coord[i+1]]
    #                CroppedMasks[idx] = mask[y_coord[j] - YShift : y_coord[j+1], x_coord[i] - XShift :x_coord[i+1]]
    #                idx +=1

    #return CroppedImgs, CroppedMasks


def cropper(image, mask, NumCropped, XCropSize, YCropSize
            ,x_coord, y_coord ,XCropNum, YCropNum, XShift, YShift):

    CroppedImgs = np.zeros((NumCropped, YCropSize, XCropSize, 3), np.uint8)
    CroppedMasks = np.zeros((NumCropped, YCropSize, XCropSize), np.uint8)
    idx = 0

    for i in range(0, XCropNum):
        for j in range(0, YCropNum):
                CroppedImgs[idx] = image[y_coord[j]:y_coord[j]+YCropSize, x_coord[i]:x_coord[i]+XCropSize]
                CroppedMasks[idx] = mask[y_coord[j]:y_coord[j]+YCropSize, x_coord[i]:x_coord[i]+XCropSize]
                idx +=1

    return CroppedImgs, CroppedMasks

def make_cropper(image_ids, images_path , masks_path, SaveCropImages, SaveCropMasks,
                 XCropSize, YCropSize, XCropCoord, YCropCoord, color = 'y',
                  shift = 0):

    if XCropCoord > XCropSize or YCropCoord > YCropSize:
        raise Exception("Not overlapping patch, the crop coord should be lesser than crop size (x or y dimension)")

    ix = shift
    flag_new_images = False

    for ax_index, name in enumerate(image_ids):
        print(name.split('.')[0])
        if color == 'y':
            if (int(name.split('.')[0]) >= 252) & (not(flag_new_images)):
                print('start cropping on new images at ids {}'.format(ix))
                dic = {}
                dic['id_new_images'] = ix
                with open('../id_new_images.pickle', 'wb') as handle:
                     pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
                flag_new_images = True

            image, mask = read_image_masks(name, images_path, masks_path)
            mask = np.squeeze(mask[:, :, 0:1])

        image, mask = read_image_masks_r(name, images_path, masks_path)
        mask = np.squeeze(mask[:, :, 0:1])

        img_height, img_width, c = image.shape

        XCropNum = int(img_width/XCropCoord)
        YCropNum = int(img_height/YCropCoord)

        NumCropped = int(img_width/XCropCoord * img_height/YCropCoord)

        # to allign the last crop with the last side parte of the image
        YShift = YCropSize - YCropCoord
        XShift = XCropSize - XCropCoord

        #x_coord = [XCropCoord*i for i in range(0, XCropNum+1)]
        #y_coord = [YCropCoord*i for i in range(0, YCropNum+1)]

        # without +1 in range(0, XCropNum +1) by construction the last crop is inside the boundary of the image
        # we can always add a last crop with corner coord
        x_coord = [XCropCoord*i for i in range(0, XCropNum)]
        y_coord = [YCropCoord*i for i in range(0, YCropNum)]

        corner_cords_x = img_width - XCropSize
        corner_cords_y = img_height - YCropSize

        if y_coord[-1] + YCropSize > img_height:
            y_coord.pop()
            y_coord.append((corner_cords_y))
        if x_coord[-1] + XCropSize > img_width:
            x_coord.pop()
            x_coord.append((corner_cords_x))

        y_coord.sort()
        x_coord.sort()

        CroppedImages, CroppedMasks = cropper(image, mask, NumCropped, XCropSize, YCropSize,
                                             x_coord, y_coord, XCropNum, YCropNum,
                                             XShift, YShift)

        for i in range(0,NumCropped):
            crop_imgs_dir = SaveCropImages + '{}_{}.tiff'.format(name.split('.')[0],i)
            crop_masks_dir = SaveCropMasks + '{}_{}.tiff'.format(name.split('.')[0],i)
            plt.imsave(fname= crop_imgs_dir, arr = CroppedImages[i])
            plt.imsave(fname= crop_masks_dir,arr = CroppedMasks[i], cmap='gray')
            ix +=1
    return



def make_weights(image_ids,  LoadMasksForWeight, SaveWeightMasks, sigma = 25, dil_k=100,
                  maximum=False, color='y'):

    if not maximum:
        max_sofar = -100000
        total = np.zeros((len(image_ids), 512, 512), dtype=np.float32)

    for ax_index, name in enumerate(image_ids):

        target = read_masks(LoadMasksForWeight, name)[:,:,0:1]
        target = target.astype(bool)
        target = remove_small_objects(target,min_size = 100)
        target = remove_small_holes(target,200)
        target = target.astype(np.uint8)*255

        tar_inv = cv2.bitwise_not(target)
        tar_dil = dilation(np.squeeze(target), selem=np.ones([dil_k, dil_k]))

        mask_sum = cv2.bitwise_and(tar_dil, tar_inv)
        mask_sum1 = cv2.bitwise_or(mask_sum, target)

        null = np.zeros((target.shape[0], target.shape[1]), dtype = np.float32)
        weighted_mask = np.zeros((target.shape[0], target.shape[1]), dtype = np.float32)

        mask, nlabels_mask = ndimage.label(target)

        if nlabels_mask < 1:

            weighted_maskk = np.ones((target.shape[0], target.shape[1]), dtype = np.float32)

        else:

            mask = remove_small_objects(mask, min_size=25, connectivity=1, in_place=False)
            mask, nlabels_mask = ndimage.label(mask)
            mask_objs = ndimage.find_objects(mask)

            for idx,obj in enumerate(mask_objs):
                new_image = np.zeros_like(mask)
                new_image[obj[0].start:obj[0].stop,obj[1].start:obj[1].stop] = mask[obj]

                new_image = np.clip(new_image, 0, 1).astype(np.uint8)
                new_image *= 255

                inverted = cv2.bitwise_not(new_image)

                distance = ndimage.distance_transform_edt(inverted)
                w = np.zeros((distance.shape[0],distance.shape[1]), dtype=np.float32)
                w1 = np.zeros((distance.shape[0],distance.shape[1]), dtype=np.float32)

                for i in range(distance.shape[0]):

                    for j in range(distance.shape[1]):

                        if distance[i, j] != 0:

                            w[i, j] = 1.*np.exp((-1 * (distance[i,j]) ** 2) / (2 * (sigma ** 2)))

                        else:

                            w[i, j] = 1

                weighted_mask = cv2.add(weighted_mask, w, mask = mask_sum)

            # Complete from inner to edge with 1.5 as weight
            weighted_mask = np.clip(weighted_mask, 1, weighted_mask.max())

            mul = target*1.5/255
            mul = mul.astype(np.float32)
            mul = np.clip(mul,1,mul.max())

            weighted_maskk = cv2.multiply(weighted_mask, mul)

        if not maximum:
            max_temp = weighted_maskk.max()
            if max_temp > max_sofar:
                max_sofar = max_temp
            print('{} on {} ({})'.format(ax_index, len(total), name))
            print('maximum so far {}'.format(max_sofar))
            total[ax_index] = weighted_maskk

        else:
            if (weighted_maskk.max()/(maximum+0.0001))> 1:
                break
            weighted_maskk = weighted_maskk*1/maximum
            target = np.clip(target, 0 , 1)
            final_target = np.dstack((target, weighted_maskk, null))

            mask_dir = SaveWeightMasks + '{}'.format(name)
            print('saving {}'.format(name))
            plt.imsave(fname=mask_dir,arr = final_target)

    if not maximum:
        np.save('total_{}.npy'.format(color), total)
        dic = {}
        dic['max_weight'] = total.max()
        with open('max_weight_{}_{}.pickle'.format(sigma, color), 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def make_cropperSS(image_ids, images_path , masks_path, SaveCropImages, SaveCropMasks,
                 XCropSize, YCropSize, XCropCoord, YCropCoord, color = 'y',
                  shift = 0):

    if XCropCoord > XCropSize or YCropCoord > YCropSize:
        raise Exception("Not overlapping patch, the crop coord should be lesser than crop size (x or y dimension)")

    ix = shift
    flag_new_images = False

    for ax_index, name in enumerate(image_ids):
        print(name.split('.')[0])
        if color == 'y':
            if (int(name.split('.')[0]) >= 252) & (not(flag_new_images)):
                print('start cropping on new images at ids {}'.format(ix))
                dic = {}
                dic['id_new_images'] = ix
                with open('../id_new_images.pickle', 'wb') as handle:
                     pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
                flag_new_images = True


        image, mask = read_image_masks_r(name, images_path, masks_path)
        mask = np.squeeze(mask[:, :, 0:1])

        img_height, img_width, c = image.shape

        XCropNum = int(img_width/XCropCoord)
        YCropNum = int(img_height/YCropCoord)

        NumCropped = int(img_width/XCropCoord * img_height/YCropCoord)

        # to allign the last crop with the last side parte of the image
        YShift = YCropSize - YCropCoord
        XShift = XCropSize - XCropCoord

        #x_coord = [XCropCoord*i for i in range(0, XCropNum+1)]
        #y_coord = [YCropCoord*i for i in range(0, YCropNum+1)]

        # without +1 in range(0, XCropNum +1) by construction the last crop is inside the boundary of the image
        # we can always add a last crop with corner coord
        x_coord = [XCropCoord*i for i in range(0, XCropNum)]
        y_coord = [YCropCoord*i for i in range(0, YCropNum)]

        corner_cords_x = img_width - XCropSize
        corner_cords_y = img_height - YCropSize

        if y_coord[-1] + YCropSize > img_height:
            y_coord.pop()
            y_coord.append((corner_cords_y))
        if x_coord[-1] + XCropSize > img_width:
            x_coord.pop()
            x_coord.append((corner_cords_x))

        y_coord.sort()
        x_coord.sort()

        CroppedImages, CroppedMasks = cropperSS(image, mask, NumCropped, XCropSize, YCropSize,
                                             x_coord, y_coord, XCropNum, YCropNum,
                                             XShift, YShift)

        for i in range(0,NumCropped):
            ret, img_y = cv2.threshold(CroppedMasks[i], 75, 255, cv2.THRESH_BINARY)
            img_y = img_y.astype(bool)
            if np.sum(img_y==1) > ((CroppedMasks[i].size)//16):
                crop_imgs_dir = SaveCropImages + '{}_{}_label1label.tiff'.format(name.split('.')[0],i)
                crop_masks_dir = SaveCropMasks + '{}_{}_label1label.tiff'.format(name.split('.')[0],i)
                plt.imsave(fname= crop_imgs_dir, arr = CroppedImages[i])
                plt.imsave(fname= crop_masks_dir,arr = img_y, cmap='gray')
                ix +=1
            else:
                if np.random.random(1) > 0.95:
                    crop_imgs_dir = SaveCropImages + '{}_{}_label0label.tiff'.format(name.split('.')[0],i)
                    crop_masks_dir = SaveCropMasks + '{}_{}_label0label.tiff'.format(name.split('.')[0],i)
                    plt.imsave(fname= crop_imgs_dir, arr = CroppedImages[i])
                    plt.imsave(fname= crop_masks_dir,arr = img_y, cmap='gray')
                    ix +=1
    return

def cropperSS(image, mask, NumCropped, XCropSize, YCropSize
            ,x_coord, y_coord ,XCropNum, YCropNum, XShift, YShift):

    CroppedImgs = np.zeros((NumCropped, YCropSize, XCropSize, 3), np.uint8)
    CroppedMasks = np.zeros((NumCropped, YCropSize, XCropSize), np.uint8)
    idx = 0

    for i in range(0, XCropNum):
        for j in range(0, YCropNum):
                CroppedImgs[idx] = image[y_coord[j]:y_coord[j]+YCropSize, x_coord[i]:x_coord[i]+XCropSize]
                CroppedMasks[idx] = mask[y_coord[j]:y_coord[j]+YCropSize, x_coord[i]:x_coord[i]+XCropSize]
                idx +=1

    return CroppedImgs, CroppedMasks