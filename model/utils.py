import torch
from torch import nn
from dataset_loader.image_loader import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms as T
import torch.nn.functional as F
import numpy as np
import math
import sys
sys.path.append('..')
from config import *
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

class Dec1(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dec1, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class LinConstr(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinConstr, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class ConstrainedDec(nn.Linear):
    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return F.conv2d(input, self.weight.clamp(min=-1*10**6, max=1*10**6), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def load_data_train_eval(batch_size=16, validation_split=0.3,images_path=AugCropImagesBasic,
                         masks_path=AugCropMasksBasic, grayscale=False, num_workers=0,
                         shuffle_dataset=True, random_seed=42, ngpus=1, ae=None):

    #root = os.getcwd()
    #root = Path(root).as_posix()
    #if ae == "ae":
    #    root = root + '/DATASET/train_val/crop_augmented_AE/'
    #else:
    #    root = root + '/DATASET/train_val/crop_augmented/'

    transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                           T.ToTensor()
                          #T.Lambda(lambda x: x.permute(2, 0, 1))
                           ])
    cells_images = CellsLoader(images_path, masks_path, val_split=0.3, grayscale=grayscale, transform=transform, ae=ae)

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
def WeightedNLLLoss(zero_weight, one_weight):

    def weighted_NLL(mu, sigma, y):

        au = 0.5*torch.log(2*np.pi*(sigma*sigma)) #aleatoric uncertainty
        ne = (torch.square(mu - y))/(2*torch.square(sigma))#normalized error
        nll_loss = au + ne

        # Apply the weights
        class_weight_vector = y[:,0:1,:,:] * one_weight + (1. - y[:,0:1,:,:]) * zero_weight

        weight_vector = class_weight_vector * y[:,1:2,:,:]
        weighted_nll = weight_vector * nll_loss

        # Return the mean error
        return torch.mean(weighted_nll)

    return weighted_NLL

def loss_VAE(mu, sigma, x):

    au = 0.5*torch.log(2*np.pi*(sigma*sigma)) #aleatoric uncertainty
    ne = (torch.square(mu - x))/(2*torch.square(sigma))#normalized error
    nll_loss = au + ne
    sigma_mean = torch.mean(sigma, axis=1)
    #return torch.mean(torch.sum(nll_loss, dim=[-1.-2])), au, ne
    return torch.mean(nll_loss), torch.mean(au, axis=1), torch.mean(ne, axis=1), sigma_mean


def gaussian_loss(par1, par2, y):
    # Gaussian distributed variables
    mu = par1
    sigma = par2
    norm_y = torch.div(y - mu, sigma)
    single_NLL = torch.log(sigma) + 0.5*torch.square(norm_y)
    #nll_loss = torch.sum(single_NLL, axis=-1)
    nll_loss = torch.sum(single_NLL, axis=(-1,-2))
    return torch.sum(nll_loss)

def KL_loss_forVAE(mu, sigma):
    mu_prior = torch.tensor(0)
    sigma_prior = torch.tensor(1)
    kl_loss = torch.mul(torch.mul(sigma, sigma), torch.mul(sigma_prior,sigma_prior))
    div = torch.div(mu_prior - mu, sigma_prior)
    kl_loss += torch.mul(div, div)
    kl_loss += torch.log(torch.div(sigma_prior, sigma)) -1
    #kl_loss = torch.sum(kl_loss, axis=(-1,-2))
    ##return 0.5 * torch.sum(kl_loss)
    return 0.5 * torch.mean(kl_loss)

