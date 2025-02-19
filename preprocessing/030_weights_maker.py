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

from config import *
from utils import *

image_ids = os.listdir(CropMasks)
image_ids.sort()
ix = [int(x.split('.')[0]) for x in image_ids]
ix.sort()
image_ids = [str(x)+'.tiff' for x in ix]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='weighting masks')

    parser.add_argument('--CropMasks', nargs="?", default = CropMasks, help='path including mask to weight')
    parser.add_argument('--CropWeightedMasks', nargs="?", default = CropWeightedMasks, help='path where save weighted mask')
    parser.add_argument('--save_images_path', nargs="?", default = CropImages, help='save images path')

    parser.add_argument('--normalize', action='store_const', const=True, default=False, help='find the maximum for normalization filling the total array in 030_weights_maker.py')
    parser.add_argument('--start_from_zero', action='store_const', const=True, default=False, help='delete all file in destination folder')
    parser.add_argument('--continue_after_normalization', action='store_const', const=True, default=False,  help='after finding the maximum continue with the weighted maps creation')
    parser.add_argument('--resume_after_normalization', action='store_const', const=True, default=True, help='If you stopped the weight maker after finding the max and you want to resum only the second step of the process')

    parser.add_argument('--maximum', nargs="?", type = int, default = 3.8177538,  help='Maximum value for normalization')
    parser.add_argument('--sigma', nargs="?", type = int, default = 25,  help='kernel for cumpling cell deacaying influence')

    args = parser.parse_args()

    if args.start_from_zero:
        print('deleting existing files in destination folder')
        try:
            shutil.rmtree(CropWeightedMasks)
        except:
            pass
        os.makedirs(CropWeightedMasks, exist_ok=True)

        print('start new weighting mask')

    # First step: find the maximum weight value
    if args.normalize:
        make_weights(image_ids,  args.CropMasks, args.CropWeightedMasks, sigma = args.sigma, maximum=False)
    # Second step following the first step: use the previous value to normalize the final weighted masks
        if args.continue_after_normalization: 
            with open('max_weight_{}.pickle'.format(sigma), 'rb') as handle:
                dic = pickle.load(handle)
            maximum = dic['max_weight']
            make_weights(image_ids,  args.CropMasks, args.CropWeightedMasks, sigma = args.sigma, maximum=maximum)
    # Only the second step: you already get the maximum weight value 
    elif args.resume_after_normalization:
        try:
            with open('max_weight_{}.pickle'.format(sigma), 'rb') as handle:
                dic = pickle.load(handle)
            maximum = dic['max_weight']
            make_weights(image_ids,  args.CropMasks, args.CropWeightedMasks, sigma = args.sigma, maximum=maximum)
        except:
            print('useing default value')
            already_done = len(os.listdir(args.CropWeightedMasks))
            ids = range(already_done, len(image_ids))

            image_ids_continue = image_ids[already_done:]

            make_weights(image_ids_continue, args.CropMasks, args.CropWeightedMasks, sigma=args.sigma, maximum=args.maximum)

    # Only the second step: you already get the maximum weight value and this is the default value passed with the args.maximum arguments
    else:
        make_weights(image_ids,  args.CropMasks, args.CropWeightedMasks, sigma = args.sigma, maximum=args.maximum)
