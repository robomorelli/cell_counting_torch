from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from config import *

def find_centers(target):

    pred_label, pred_count = ndimage.label(target)
    pred_objs = ndimage.find_objects(pred_label)

    # compute centers of predicted objects
    pred_centers = []
    for ob in pred_objs:
        pred_centers.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                             (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))
    return pred_centers

def create_mask(pred_centers, filename, radius=10, h=512, w=512):

    mask = np.zeros((h, w, 3), np.uint8)
    mask.fill(0)
    print('\nGenerating mask for folder {}\n'.format(filename))

    for y, x in pred_centers:
        #coords = [float(x.strip('(),')) for x in coords.split()]
        cv2.circle(mask, (int(x), int(y)), radius, [255, 255, 255], -1)
    return mask


def model_inference(data_loader, model, vae_flag = False, device='cpu', split=40):
    model.eval()
    preds = []
    targets = []
    images = []
    if vae_flag:
        for ix, (x, y) in enumerate(data_loader):
            print("batch {} on {}".format(ix, len(data_loader)))
            with torch.no_grad():
                mu, sigma, segm, (mu_p, sigma_p) = model(x.to(device))
                #recon_loss, au, ne, au_1ch, ne_1ch = loss_VAE_rec(mu_p.to(device), sigma_p.to(device), x.to(device))
                preds.extend(segm.detach().cpu())
                images.extend(x)
                targets.extend(y)
                torch.cuda.empty_cache()

    else:
        for ix, (x, y) in enumerate(data_loader):
            with torch.no_grad():
                print("batch {} on {}".format(ix, len(data_loader)))
                results = model(x.to(device)).cpu().detach()
                preds.extend(results)
                images.extend(x)
                targets.extend(y)
                torch.cuda.empty_cache()

    return images, targets, preds
