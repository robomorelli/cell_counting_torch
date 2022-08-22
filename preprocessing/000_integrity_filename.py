import argparse
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from config import *

from config import *

'''
This script is used to format to .tiff all the images. Red dataset initially have different format mixed (TIF, tif, JPG) among 
the images, is better to convert to .tiff. The masks have instead "_mask" as suffix of the image name but no itnervention is taken for this
'''
IMG_CHANNELS = 3

def main(args):

    AllImages = args.image_path
    AllMasks = (Path(args.image_path).parent.parent).as_posix() + '/all_masks/masks/'

    images_name = os.listdir(AllImages)
    masks_name = os.listdir(AllMasks)

    for ix, im_name in enumerate(images_name):
        #############Processing###############
        print(im_name)
        img_x = cv2.imread(str(AllImages) + im_name)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_dir = AllImages + '{}'.format(im_name.split('.')[0] + '.tiff')
        plt.imsave(fname=img_dir, arr=np.squeeze(img_x))

    for ix, mask_name in enumerate(masks_name):
        #############Processing###############
        print(mask_name)
        img_y = cv2.imread(str(AllMasks) + mask_name)
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]
        mask_dir = AllMasks + '{}'.format(mask_name.split('_')[0] + '.tiff')
        plt.imsave(fname=mask_dir, arr=np.squeeze(img_y), cmap='gray')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--image_path', nargs="?", default=AllImagesR,
                        help='the folder including the masks to crop')
    args = parser.parse_args()

    main(args)
