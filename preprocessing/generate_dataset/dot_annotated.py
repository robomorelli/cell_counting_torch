import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import shutil
import random
from config import *
from preprocessing.generate_dataset.utils import *
from dataset_loader.image_loader import CellsLoader
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
import torch.nn as nn
from model.resunet import *

np.random.seed(0)

def main(args):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    try:
        shutil.rmtree(AugDotAnnotatedImages)
    except:
        pass
    os.makedirs(AugDotAnnotatedImages, exist_ok=True)
    try:
        shutil.rmtree(AugDotAnnotatedMasks)
    except:
        pass
    os.makedirs(AugDotAnnotatedMasks, exist_ok=True)
    cropped_nams = os.listdir(AugCropImages.replace('preprocessing', ''))
    ix = 0
    for im_names in cropped_nams:
        #take masks and center of it
        mask = cv2.imread(str(AugCropMasks).replace('preprocessing', '') + '/' + im_names)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)[:, :, 0:1]
        centers = find_centers(mask)
        #place a dot annonation on the center
        mask = create_mask(centers, im_names, args.radius)
        #save the image
        plt.imsave(os.path.join(str(AugDotAnnotatedMasks).replace('preprocessing', ''), im_names.split('.')[0]+'.png'), mask)

def refine(args):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    try:
        shutil.rmtree(AugDotAnnotatedRefinedMasks.replace('preprocessing', ''))
    except:
        pass
    os.makedirs(AugDotAnnotatedRefinedMasks.replace('preprocessing', ''), exist_ok=True)

    transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                               T.ToTensor()])
    cells_images = CellsLoader(AugCropImages.replace('preprocessing', ''), AugDotAnnotatedMasks.replace('preprocessing', ''),
                               val_split=0.3, transform=transform)
    data_loader = DataLoader(cells_images, batch_size=8, shuffle=False)
    filenames = cells_images.imgs_list

    added_path = 'dot_annotated/{}/{}/'.format(args.dataset, args.model_name)
    resume_path = ModelResults + added_path + args.model_name + '.h5'
    model = load_model(resume_path=resume_path.replace('preprocessing', ''), device=device, n_features_start=16, n_out=1)

    # model inference
    for ix, f in enumerate(filenames):
        img = cv2.imread(AugCropImages.replace('preprocessing', '') + f)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.float()

        with torch.no_grad():
            print("image {} num {} ".format(f, ix))
            result = model(img.to(device)).cpu().detach()
            torch.cuda.empty_cache()

        #images, targets, preds = model_inference(data_loader, model)
        th = 0.7
        preds_t = (np.squeeze(result[0:1, :, :]) > th)

        #save the image
        #for im_names in filenames:
        plt.imsave(os.path.join(str(AugDotAnnotatedRefinedMasks).replace('preprocessing', ''), f.split('.')[0]+'.png'), preds_t)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--radius', nargs="?", default=12, help='the folder including the masks to crop')
    parser.add_argument('--dataset', nargs="?", default='yellow', help='the folder including the masks to crop')
    parser.add_argument('--model_name', nargs="?", default='c-resunet_y', help='model_name')
    args = parser.parse_args()

    refine(args)


