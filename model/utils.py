import torch
from torch import nn
from dataset_loader.image_loader import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms as T
import numpy as np
from pathlib import Path

clip_x_to0 = 1e-4

def SmashTo0(x):
    return 0*x

class InverseSquareRootLinearUnit(nn.Module):

    def __init__(self, min_value=5e-3):
        super(InverseSquareRootLinearUnit, self).__init__()
        self.min_value = min_value

    def forward(self, x):
        return 1. + self.min_value \
               + torch.where(torch.gt(x, 0), x, torch.div(x, torch.sqrt(1 + (x * x))))

class ClippedTanh(nn.Module):

    def __init__(self, min_value=5e-3):
        super(ClippedTanh, self).__init__()

    def forward(self, x):
        return 0.5 * (1 + 0.999 * torch.tanh(x))

class SmashTo0(nn.Module):

    def __init__(self):
        super(SmashTo0, self).__init__()

    def forward(self, x):
        return 0*x


def load_data_train_eval(batch_size=16, validation_split=0.3, num_workers=0,
                         shuffle_dataset=True, random_seed=42, ngpus=1, ae=None):

    root = os.getcwd()
    root = Path(root).as_posix()
    if ae == "ae":
        root = root + '/DATASET/train_val/crop_augmented_AE/'
    else:
        root = root + '/DATASET/train_val/crop_augmented/'
    transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                           T.ToTensor()
                          #T.Lambda(lambda x: x.permute(2, 0, 1))
                           ])
    cells_images = CellsLoader(root + "images/", root + "masks/", val_split=0.3, transform=transform, ae=ae)

    dataset_size = len(cells_images)
    print('dataset size is {}'.format(dataset_size))
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

        #we should first make a sum reduction on rows and column and then take a mean over element of batch
        #s1 = torch.sum(weighted_b_ce, axis=-2)
        #s2 = torch.sum(s1, axis=-1)

        # Return the mean error
        return torch.mean(weighted_b_ce)

    return weighted_binary_crossentropy


#def VAE_loss(bce_loss, mu, logvar):
    #    """
    #    This function will add the reconstruction loss (BCELoss) and the
    #    KL-Divergence.
    #    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #    :param bce_loss: recontruction loss
    #    :param mu: the mean from the latent vector
    #    :param logvar: log variance from the latent vector
    #    """
    #    BCE = bce_loss
    #    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#    return BCE + KLD

def loss_VAE(mu, sigma, y):

    au = 0.5*torch.log(2*np.pi*(sigma*sigma)) #aleatoric uncertainty
    ne = (torch.square(mu - y))/(2*torch.square(sigma))#normalized error
    nll_loss = au + ne

    return torch.mean(nll_loss), au, ne

