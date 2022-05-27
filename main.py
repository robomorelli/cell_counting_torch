import random as rn
import numpy as np
from model.resunet import *
from model.utils import EarlyStopping, WeightedLoss, load_data_train_eval
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

np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(123456)

def train():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print('added visible gpu')
        ngpus = torch.cuda.device_count()

    else:
        ngpus=1

    num_workers = 12
    train_loader, validation_loader = load_data_train_eval(batch_size=16, validation_split=0.3,
                        num_workers=num_workers, shuffle_dataset=True, random_seed=42, ngpus=ngpus)

    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start = 16, n_out = 1,
                                      pretrained = False, progress= True).to(device))

    #Train Loop####
    """
    Set the model to the training mode first and train
    """
    val_loss = 10 ** 16
    patience = 5
    lr = 0.003
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  factor=0.8, patience=patience, threshold=0.0001,
                                              threshold_mode='rel', cooldown=0, min_lr=9e-8, verbose=True)
    early_stopping = EarlyStopping(patience=7)
    epochs = 200
    model_name = 'c-resunt_v4.h5'
    Wloss = WeightedLoss(1, 1.5)
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                # .permute(0, 3, 1, 2).float()
                x = model(data[0].to(device))
                y = data[1].to(device)
                loss = Wloss(x, y)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            ###############################################
            # eval mode for evaluation on validation dataset_loader
            ###############################################
            with torch.no_grad():
                model.eval()
                temp_val_loss = 0
                with tqdm(validation_loader, unit="batch") as vepoch:
                    for i, data in enumerate(vepoch):
                        optimizer.zero_grad()
                        x = model(data[0].to(device))
                        y = data[1].to(device)
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
