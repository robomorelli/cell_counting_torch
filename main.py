import random as rn
import numpy as np
from dataset.image_loader import *
from model.resunet import *
from model.utils import EarlyStopping, WeightedLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms as T
import torch.nn as nn
from pathlib import Path
import torch.multiprocessing as mp
from tqdm import tqdm
from time import sleep

import fastai

np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

def train():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print('added visible gpu')
        ngpus = torch.cuda.device_count()

    batch_size = 8
    validation_split = 0.3
    shuffle_dataset = True
    random_seed = 42

    root = "/davinci-1/home/morellir/artificial_intelligence/repos/cells_torch/DATASET/train_val/crop_augmented/"
    transform = T.Compose([T.Lambda(lambda x: x*1./255)])
    cells_images = CellsLoader(root + "images/", root + "masks/", val_split=0.3, transform = transform)

    dataset_size = len(cells_images)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices, )
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(cells_images, batch_size=batch_size*ngpus,
                              sampler=train_sampler)
    validation_loader = DataLoader(cells_images, batch_size=batch_size*ngpus,
                                   sampler=valid_sampler)

    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start = 16, n_out = 1,
                                      pretrained = False, progress= True).to(device))

    #Train Loop####
    """
    Set the model to the training mode first and train
    """
    val_loss = 10 ** 16
    patience = 1
    lr = 0.003
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  factor=0.8, patience=patience, threshold=0.0001,
                                              threshold_mode='rel', cooldown=0, min_lr=9e-8, verbose=True)
    early_stopping = EarlyStopping(patience=7)
    epochs = 100
    model_name = 'c-resunet.h5'
    Wloss = WeightedLoss(1, 1.5)
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                x = model(data[0].permute(0, 3, 1, 2).float().to(device))
                y = data[1].permute(0, 3, 1, 2).float().to(device)
                loss = Wloss(x, y)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)
                #if i % 1 == 0:
                #    print("Loss: {} batch {} on total of {}".format(loss.item(), i, len(train_loader)))

            ###############################################
            # eval mode for evaluation on validation dataset
            ###############################################
            with torch.no_grad():
                model.eval()
                temp_val_loss = 0
                with tqdm(validation_loader, unit="batch") as vepoch:
                    for i, data in enumerate(vepoch):
                        optimizer.zero_grad()
                        x = model(data[0].permute(0, 3, 1, 2).float().to(device))
                        y = data[1].permute(0, 3, 1, 2).float().to(device)
                        loss = Wloss(x, y)
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
                        torch.save(model.state_dict(), save_model_path / model_name)
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
    train()
