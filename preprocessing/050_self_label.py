import argparse
import sys
sys.path.append('../')
from tqdm import tqdm
import torch
import numpy as np
from torchvision.transforms import transforms as T
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.append('..')
sys.path.append('../dataset_loader')
sys.path.append('../model')
from config import *
from dataset_loader.image_loader import *
from model.resunet import *
from utils import *
from skimage.morphology import remove_small_holes, remove_small_objects,\
label, erosion, dilation, local_maxima, skeletonize, binary_erosion, remove_small_holes

def main(args):
    images_name = os.listdir(args.images_path)
    # define device to load the model and make inference
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # the model came from autencoder with pre layer for binary segmantion
    resume_path = args.model_results + args.model_name +'.h5'
    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=1).to(device))
    layers_to_remove = ['module.head.conv2d.weight', 'module.head.conv2d.bias']
    layers_to_rename = ['module.head.conv2d_binary.weight', 'module.head.conv2d_binary.bias']
    checkpoint_file = torch.load(resume_path)
    for k in list(checkpoint_file.keys()):
        if k in layers_to_remove:
            checkpoint_file.pop(k)
    for k in list(checkpoint_file.keys()):
        if k in layers_to_rename:
            checkpoint_file[k.replace("_binary", "")] = checkpoint_file.pop(k)
    model.load_state_dict(checkpoint_file, strict=False)
    # load images to make the inference
    transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                           T.ToTensor(),
                           ])
    cells_images = CellsLoader(args.images_path, transform=transform, unlabelled=True)
    test_loader = DataLoader(cells_images, batch_size=16, shuffle=False)

    with torch.no_grad():
        ix = 0
        model.eval()
        with tqdm.tqdm(test_loader, unit="batch") as tepoch:
            for i, image in enumerate(tepoch):
                x = image.to(device)
                out = model(x)
                out_cpu = out.cpu().numpy()
                out_cpu_th = np.where(out_cpu > args.threshold, 255, 0)
                for image in out_cpu_th:
                    image = image.astype(bool)

                    if '_' in images_name[ix]:
                        if int(images_name[ix].split('_')[-1].split('.')[0]) > 3 and int(images_name[ix].split('_')[0]) < 2044:
                            image = np.zeros_like(image)
                        else:
                            print('erosion')
                            image = erosion(np.squeeze(image), selem=np.ones([args.dil_k, args.dil_k]))
                            image = remove_small_objects(image, min_size=100)
                            image = remove_small_holes(image, 200)
                            image = image.astype(np.uint8) * 255
                    else:
                        #image = erosion(np.squeeze(image), selem=np.ones([args.dil_k, args.dil_k]))
                        image = remove_small_objects(image, min_size=100) #100 before
                        #image = remove_small_holes(image, 200)
                        image = image.astype(np.uint8) * 255

                        #image_normal_and_big = dilation(np.squeeze(image), selem=np.ones([args.dil_k, args.dil_k]))
                        #image_normal_and_big = remove_small_objects(image_normal_and_big, min_size=30)
                        #image_normal_and_big = remove_small_holes(image_normal_and_big , 200)
                        #image_normal_and_big = image_normal_and_big.astype(np.uint8) * 255

                        #image_big = dilation(np.squeeze(image), selem=np.ones([args.dil_k, args.dil_k])) # to connect fragmentens inside big artifact
                        #image_big = remove_small_objects(image_big, min_size=200)
                        #image_big = remove_small_holes(image_big , 200)
                        #image_big = image_big .astype(np.uint8) * 255

                        #image = image_normal_and_big - image_big
                        #image = erosion(np.squeeze(image), selem=np.ones([args.dil_k, args.dil_k]))


                    plt.imsave(args.masks_path + images_name[ix], np.squeeze(image), cmap='gray')
                    ix += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--images_path', nargs="?", default=AuCropImages, help='the folder including the images to predict')
    parser.add_argument('--masks_path', nargs="?", default=CropMasksSS, help='folder where to save the predicted images')
    parser.add_argument('--model_results', nargs="?", default=ModelResults, help='folder of trained models')
    parser.add_argument('--model_name', nargs="?", default='c-resunet_y_ae', help='modle name to load')
    parser.add_argument('--threshold', nargs="?", default=0.8, help='threshodl to apply on the predicted images')
    parser.add_argument('--dil_k', nargs="?", default=3, help='dilation kernel')
    args = parser.parse_args()

    main(args)
