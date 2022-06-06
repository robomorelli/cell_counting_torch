import random as rn
import numpy as np
from model.resunet import *
from model.utils import EarlyStopping, WeightedLoss, load_data_train_eval, loss_VAE, \
    KL_loss_forVAE, gaussian_loss, WeightedNLLLoss
from dataset_loader.image_loader import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms as T
import torch.nn as nn
from pathlib import Path
import torch.multiprocessing as mp
from tqdm import tqdm
from time import sleep
import torch
import sys
sys.path.append('..')
from config import *



np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(123456)

def train(ae=None):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print('added visible gpu')
        ngpus = torch.cuda.device_count()

    else:
        ngpus=1

    num_workers = 0
    bs = 16
    if ae == 'ae':
        n_out=3

    train_loader, validation_loader = load_data_train_eval(batch_size=bs, validation_split=0.3,images_path=AugCropImages
                                                           , masks_path=AugCropMasks, grayscale = False,
                                                           num_workers=num_workers, shuffle_dataset=True,
                                                           random_seed=42, ngpus=ngpus, ae=ae)

    model_name = 'hydra_noLongConn_hpc.h5'
    resume = False

    if resume:
        resume_path = ModelResults + model_name
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            if device == 'cpu':
                checkpoint = torch.load(resume_path)
            else:
                # Map model to be loaded to specified single gpu.
                model = nn.DataParallel(c_resunetVAE(arch='c-ResUnetVAE', n_features_start=16, n_out=1, n_outRec=1,
                                                     pretrained=False, progress=True)).to(device)
                model.load_state_dict(torch.load(resume_path))
                #checkpoint = torch.load(args.resume, map_location=loc)
            #args.start_epoch = checkpoint['epoch']
            #best_acc1 = checkpoint['best_acc1']
            #if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
            #    best_acc1 = best_acc1.to(args.gpu)
            #model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            #scheduler.load_state_dict(checkpoint['scheduler'])
            #print("=> loaded checkpoint '{}' (epoch {})"
            #      .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))
            model = nn.DataParallel(c_resunetVAE(arch='c-ResUnetVAE', n_features_start=16, zDIm=64, n_out=1, n_outRec=1,
                                      pretrained=False, resume=model_name, progress=True, device=device).to(device))
    else:
        model = nn.DataParallel(c_resunetVAE(arch='c-ResUnetVAE', n_features_start=16, zDIm=64, n_out=1, n_outRec=1,
                                             pretrained=False, resume=model_name, progress=True, device=device).to(device))

    #Train Loop####
    """
    Set the model to the training mode first and train
    """
    val_loss = 10 ** 16
    patience = 5
    patience_lr = 3
    lr = 0.003
    epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  factor=0.8, patience=patience_lr, threshold=0.0001,
                                              threshold_mode='rel', cooldown=0, min_lr=9e-8, verbose=True)
    early_stopping = EarlyStopping(patience=patience)

    loss_type='weighted'
    if loss_type=='weighted':
        criterion = WeightedLoss(1, 1.5)

    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (image, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                y = target.to(device)
                x = image.to(device)
                gray_RGB, mu, sigma, segm, (mu_p, sigma_p) = model(x)
                #mu, sigma, segm, (mu_p, sigma_p) = model(x)

                segm_loss = criterion(segm, y)
                #recon_loss, au, ne = loss_VAE(mu_p, sigma_p, x)
                recon_loss, au, ne = loss_VAE(mu_p, sigma_p, gray_RGB)

                #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                #KLD = -0.5 * torch.sum(1 + torch.log(sigma*sigma) - mu.pow(2) - sigma*sigma)
                scale = 10
                kld_factor = 0.6
                KLD = KL_loss_forVAE(mu, sigma).mean()
                loss = recon_loss + kld_factor*KLD + scale*segm_loss

                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item(), nll = recon_loss.item(), KLD=KLD.item()*kld_factor, segm_loss=scale*segm_loss.item())
            ###############################################
            # eval mode for evaluation on validation dataset_loader
            ###############################################
            with torch.no_grad():
                model.eval()
                temp_val_loss = 0
                with tqdm(validation_loader, unit="batch") as vepoch:
                    for i, (image, target) in enumerate(vepoch):
                        optimizer.zero_grad()

                        y = target.to(device)
                        x = image.to(device)
                        gray_RGB, mu, sigma, segm, (mu_p, sigma_p) = model(x)
                        #mu, sigma, segm, (mu_p, sigma_p) = model(x)

                        segm_loss = criterion(segm, y)
                        #recon_loss, au, ne = loss_VAE(mu_p, sigma_p, x)
                        recon_loss, au, ne = loss_VAE(mu_p, sigma_p, gray_RGB)

                        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        # KLD = -0.5 * torch.sum(1 + torch.log(sigma*sigma) - mu.pow(2) - sigma*sigma)
                        scale = 10
                        KLD = KL_loss_forVAE(mu, sigma).mean()
                        loss = recon_loss + kld_factor * KLD + scale * segm_loss

                        temp_val_loss += loss
                        if i % 10 == 0:
                            print("VALIDATION Loss: {} batch {} on total of {}".format(loss.item(),
                                                                                    i, len(validation_loader)))

                    temp_val_loss = temp_val_loss / len(validation_loader)
                    print('validation_loss {}'.format(temp_val_loss))
                    scheduler.step(temp_val_loss)
                    if temp_val_loss < val_loss:
                        print('val_loss improved from {} to {}, saving model to {}' \
                              .format(val_loss, temp_val_loss, save_model_path))
                        path_posix = (save_model_path / model_name).as_posix()
                        save_path = path_posix.split('.')[0] + '_{}'.format(epoch) + '.h5'
                        torch.save(model.state_dict(), save_path)
                        #torch.save(model.state_dict(), save_model_path / model_name)
                        val_loss = temp_val_loss
                    early_stopping(temp_val_loss)
                    if early_stopping.early_stop:
                        break

if __name__ == "__main__":
    ###############################################
    # TO DO: add parser for parse command line args
    ###############################################
    save_model_path = Path('./model_results_torch')
    if not (save_model_path.exists()):
        print('creating path')
        os.makedirs(save_model_path)
    ae = None #['ae', 'vae']
    train(ae)
