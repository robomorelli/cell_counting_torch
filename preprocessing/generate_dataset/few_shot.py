import os

import pandas as pd
import numpy as np
import argparse
import os
import shutil
import random
from config import *

np.random.seed(0)

def main(args):

    if args.color=='red':
        im_names = os.listdir(TrainValImagesR)
        try:
            shutil.rmtree(AugCropImagesFewShotR)
        except:
            pass
        os.makedirs(AugCropImagesFewShotR, exist_ok=True)
        try:
            shutil.rmtree(AugCropMasksFewShotR)
        except:
            pass
        os.makedirs(AugCropMasksFewShotR, exist_ok=True)

        if args.few_shot != 'all':
            indexes = np.random.randint(0, len(im_names), args.few_shot)
        else:
            indexes = np.arange(0, len(im_names))

        to_sample_root = [im_names[i] for i in indexes]

        cropped_nams = os.listdir(AugCropImagesBasicR)
        for im_names in cropped_nams:
            if '_'.join(im_names.split('_')[:-2]) + '.tiff' in to_sample_root:
                shutil.copy(os.path.join(AugCropImagesBasicR,im_names), os.path.join(AugCropImagesFewShotR, im_names))
                shutil.copy(os.path.join(AugCropMasksBasicR,im_names), os.path.join(AugCropMasksFewShotR, im_names))

    if 'yellow' in args.color:
        im_names = os.listdir(TrainValImages)
        try:
            shutil.rmtree(AugCropImagesFS)
        except:
            pass
        os.makedirs(AugCropImagesFS, exist_ok=True)
        try:
            shutil.rmtree(AugCropMasksFS)
        except:
            pass
        os.makedirs(AugCropMasksFS, exist_ok=True)

        #indexes = np.random.randint(0, len(im_names), args.few_shot)
        #to_sample_root = [im_names[i] for i in indexes]

        cropped_nams = os.listdir(AugCropImages)
        ix = 0
        for im_names in cropped_nams:
            if ix < int(args.color.split('_')[1]):
                shutil.copy(os.path.join(AugCropImages,im_names), os.path.join(AugCropImagesFS, im_names))
                shutil.copy(os.path.join(AugCropMasks,im_names), os.path.join(AugCropMasksFS, im_names))
            ix += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--color', nargs="?", default='red', help='the folder including the masks to crop')
    parser.add_argument('--few_shot', nargs="?", default=10, help='the folder including the masks to crop')
    args = parser.parse_args()

    main(args)
