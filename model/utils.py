import torch
from torch import nn

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


def create_weighted_binary_crossentropy_overcrowding(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        b_ce = K.binary_crossentropy(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])

        # Apply the weights
        class_weight_vector = y_true[:,:,:,0:1] * one_weight + (1. - y_true[:,:,:,0:1]) * zero_weight

        weight_vector = class_weight_vector * y_true[:,:,:,1:2]
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy