import torch
from torch import nn
from dataset_loader.image_loader import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms as T
import numpy as np
from pathlib import Path

def load_data_train_eval(batch_size=16, validation_split=0.3, num_workers = 0,
                         shuffle_dataset=True, random_seed=42, ngpus = 1):

    root = os.getcwd()
    root = Path(root).as_posix()
    root = root + '/DATASET/train_val/crop_augmented/'
    transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                           T.ToTensor()
                          #T.Lambda(lambda x: x.permute(2, 0, 1))
                           ])
    cells_images = CellsLoader(root + "images/", root + "masks/", val_split=0.3, transform=transform)

    dataset_size = len(cells_images)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices, )
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(cells_images, batch_size=batch_size * ngpus,
                              sampler=train_sampler, num_workers=num_workers)
    validation_loader = DataLoader(cells_images, batch_size=batch_size * ngpus,
                                   sampler=valid_sampler, num_workers=num_workers)

    return train_loader, validation_loader


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
def WeightedLoss(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_pred, y_true):

        #b_ce = nn.BCEWithLogitsLoss(reduction = 'none')(y_true[:,0:1,:,:].float(), y_pred[:,0:1,:,:].float()) #try without reduction
        b_ce = nn.BCELoss(reduction='none')(y_pred[:, 0:1, :, :].float(), y_true[:, 0:1, :, :].float())
        # Apply the weights
        class_weight_vector = y_true[:,0:1,:,:] * one_weight + (1. - y_true[:,0:1,:,:]) * zero_weight

        weight_vector = class_weight_vector * y_true[:,1:2,:,:]
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return torch.mean(weighted_b_ce)

    return weighted_binary_crossentropy
