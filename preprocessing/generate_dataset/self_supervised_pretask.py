import os
import pandas as pd
import numpy as np
import argparse
import shutil
import random
from config import *
import argparse
import sys

sys.path.append('../preprocessing')
from preprocessing.utils import *
np.random.seed(0)

def main(args):
    # take image and target
    # for each couple use a slide window and sample both instances of cells and no cells
        # if the crop is more than half filled by a cell we have a cell sample otherwise a background one
    # resize to 128 to 128 to allow a complete encoding step and embedding
    args.images_path = args.images_path.replace('preprocessing', '')
    args.masks_path = args.masks_path.replace('preprocessing', '')
    args.save_images_path = args.save_images_path.replace('preprocessing', '')
    args.save_masks_path = args.save_masks_path.replace('preprocessing', '')
    if args.pretask == 'SigVsBkg':
        if args.start_from_zero:
            print('deleting existing files in destination folder')
            try:
                shutil.rmtree(args.save_images_path)
            except:
                pass
            os.makedirs(args.save_images_path, exist_ok=True)
            try:
                shutil.rmtree(args.save_masks_path)
            except:
                pass
            os.makedirs(args.save_masks_path, exist_ok=True)
            print('start to crop')

        image_ids = os.listdir(args.images_path)
        image_ids.sort()
        if args.color == 'y':
            Number = [int(num.split('.')[0]) for num in image_ids]
            Number.sort()
            image_ids = [str(num) + '.tiff' for num in Number]

        make_cropperSS(image_ids=image_ids, images_path=args.images_path, masks_path=args.masks_path
                     , SaveCropImages=args.save_images_path, SaveCropMasks=args.save_masks_path,
                     XCropSize=args.x_size, YCropSize=args.y_size, XCropCoord=args.x_distance, YCropCoord=args.y_distance,
                     color=args.color, shift=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--pretask', nargs="?", default="SigVsBkg", help='discriminate cells crop from the background')

    parser.add_argument('--images_path', nargs="?", default=AllImagesR, help='the folder including the images')
    parser.add_argument('--masks_path', nargs="?", default=AllMasksR, help='the folder including the masks')

    parser.add_argument('--start_from_zero', action='store_const', const=True, default=False,
                        help='remove previous file in the destination folder')
    parser.add_argument('--save_images_path', nargs="?", default=SelfSuperImagesR, help='save images path')
    parser.add_argument('--save_masks_path', nargs="?", default=SelfSuperMasksR, help='save masks path')

    parser.add_argument('--x_size', nargs="?", type=int, default=128, help='width of the crop')
    parser.add_argument('--y_size', nargs="?", type=int, default=128, help='height of the crop')
    parser.add_argument('--x_distance', nargs="?", type=int, default=100,
                        help='distance beetwen cuts points on the x axis')
    parser.add_argument('--y_distance', nargs="?", type=int, default=100,
                        help='distance beetwen cuts points on the y axis')
    parser.add_argument('--color', nargs="?", default='r', help='dataset id')
    args = parser.parse_args()

    main(args)
