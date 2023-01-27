import random as rn
import numpy as np
from model.resunet import *
from model.utils import EarlyStopping, WeightedLoss, load_data_train_eval, UnweightedLoss, WeightedLossAE
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
import argparse
from pytorch_metric_learning import losses, reducers, miners, distances, regularizers
sys.path.append('..')
from config import *


np.random.seed(40)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
#rn.seed(33323)
rn.seed(3334)

def train(args):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print('added visible gpu')
        ndevices = torch.cuda.device_count()

    else:
        ndevices=1

    num_workers = 0
    if args.ae or args.ae_bin:
        n_out=3
        #args.ae = True
        #args.loss_type='unweighted'
    else:
        n_out=1

    if args.loss_type=='weighted':
        criterion = WeightedLoss(1, 1.5)
        unweighted = False
    #elif args.loss_type=='weightedAE':
        #criterion = WeightedLossAE(1, 1)
        #unweighted = False
    elif args.loss_type=='unweightedAE':
        criterion = nn.BCELoss()
        unweighted = True
    else:
        print('unweighted loss')
        criterion = nn.BCELoss()
        unweighted = True
        #criterion = UnweightedLoss(1, 1.5) #not for autoencoder but segmentation weighted only on the class type


    train_loader, validation_loader = load_data_train_eval(dataset=args.dataset,
                                    batch_size=args.bs, validation_split=0.3,
                                    grayscale = False, num_workers=num_workers,
                                    shuffle_dataset=True, random_seed=42, ngpus=ndevices,
                                    ae=args.ae, ae_bin = args.ae_bin, few_shot_merged=args.few_shot_merged,
                                    few_shot=args.few_shot,
                                    self_supervised=args.self_supervised,
                                   weakly_supervised_ae=args.weakly_supervised_ae,
                                   weakly_supervised_ae_bin=args.weakly_supervised_ae_bin,
                                   unsupervised=args.unsupervised,
                                   unweighted=unweighted)

    learning_type = [args.self_supervised, args.few_shot, args.ae, args.ae_bin, args.fine_tuning,
                     args.weakly_supervised_ae_bin,  args.weakly_supervised_ae, args.unsupervised]
    learning_name = ['self_supervised', 'few_shot', 'autoencoder','autoencoder_bin', 'fine_tuning',
                     'weakly_supervised_ae_bin', 'weakly_supervised_ae', 'unsupervised']
    learning_zip = zip(learning_type, learning_name)
    learning = None
    for lt, ln in learning_zip:
        if lt==True:
            learning = ln
    if learning == None:
        learning = 'supervised'

    print('learning type', learning)

    if args.resume or args.fine_tuning:
        resume_path = str(args.resume_path) + '/{}/{}/{}/{}'.format(learning, args.dataset, args.model_name, args.model_name + '.h5')
        args.model_name = args.new_model_name

    if args.fine_tuning:
        if args.few_shot:
            added_path = 'fine_tuning/few_shot/{}/{}/'.format(args.dataset, args.model_name)
        elif args.few_shot_merged:
            added_path = 'fine_tuning/few_shot_merged/{}/{}/'.format(args.dataset, args.model_name)
        else:
            added_path = 'fine_tuning/{}/{}/'.format(args.dataset, args.model_name)
    elif args.ae:
        added_path = 'autoencoder/{}/{}/'.format(args.dataset, args.model_name)
    elif args.ae_bin:
        added_path = 'autoencoder_bin/{}/{}/'.format(args.dataset, args.model_name)
    elif args.few_shot:
        added_path = 'few_shot/{}/{}/'.format(args.dataset, args.model_name)
    elif args.few_shot_merged:
        added_path = 'few_shot_merged/{}/{}/'.format(args.dataset, args.model_name)
    elif args.self_supervised:
        added_path = 'self_supervised/{}/{}/'.format(args.dataset, args.model_name)
    elif args.weakly_supervised_ae:
        added_path = 'weakly_supervised_ae/{}/{}/'.format(args.dataset, args.model_name)
    elif args.weakly_supervised_ae_bin:
        added_path = 'weakly_supervised_ae_bin/{}/{}/'.format(args.dataset, args.model_name)
    elif args.unsupervised:
        added_path = 'unsupervised/{}/{}/'.format(args.dataset, args.model_name)
    else:
        added_path = 'supervised/{}/{}/'.format(args.dataset, args.model_name)

    if args.monitor_epochs:
        added_path = added_path + 'epochs/'

    if os.path.exists(args.save_model_path + added_path):
        print(" path exists", args.save_model_path + added_path)
    else:
        os.makedirs(args.save_model_path + added_path)

    if args.self_supervised:
        model = nn.DataParallel(c_resunet_enc('c-ResUnetEnc', n_features_start=16, code_dim=16,
                                          pretrained=False, resume=args.model_name, progress=True,
                                          device=device).to(device))

        val_loss = 10 ** 16
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=args.patience_lr,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=9e-8, verbose=True)
        early_stopping = EarlyStopping(patience=args.patience)

        miners_func = miners.BatchEasyHardMiner(
            pos_strategy=miners.BatchEasyHardMiner.HARD,  # HARD
            neg_strategy=miners.BatchEasyHardMiner.EASY)  # SEMIHARD
        #regularizer = regularizers.RegularFaceRegularizer()
        # loss_func = losses.NTXentLoss(temperature=0.07)
        loss_func = losses.TripletMarginLoss(margin=0.3)

        for epoch in range(args.epochs):
            model.train()
            with tqdm(train_loader, unit="batch") as tepoch:
                for i, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    optimizer.zero_grad()
                    x, lbls = batch
                    lbls = torch.squeeze(lbls)
                    embeddings = model(x)
                    lbls = lbls.type(torch.LongTensor).to(lbls.device)
                    indices_tuple = miners_func(embeddings, lbls)
                    loss = loss_func(embeddings, lbls, indices_tuple)
                    loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.item())

                with torch.no_grad():
                    model.eval()
                    temp_val_loss = 0
                    with tqdm(validation_loader, unit="batch") as vepoch:
                        for i, batch in enumerate(vepoch):
                            optimizer.zero_grad()
                            x, lbls = batch
                            lbls = torch.squeeze(lbls)
                            embeddings = model(x)
                            lbls = lbls.type(torch.LongTensor).to(lbls.device)
                            indices_tuple = miners_func(embeddings, lbls)
                            loss = loss_func(embeddings, lbls, indices_tuple)
                            temp_val_loss += loss
                            if i % 10 == 0:
                                print("VALIDATION Loss: {} batch {} on total of {}".format(loss.item(),
                                                                                        i, len(validation_loader)))

                        temp_val_loss = temp_val_loss / len(validation_loader)
                        ix, pos, _, neg, = indices_tuple
                        pdist = torch.nn.CosineSimilarity()
                        pos_sim = torch.mean((pdist(embeddings[ix], embeddings[pos]) + 1) / 2)
                        neg_sim = torch.mean((pdist(embeddings[ix], embeddings[neg]) + 1) / 2)

                        print('validation_loss {}'.format(temp_val_loss))
                        print('pos distance {} and negative distance {}'.format(pos_sim, neg_sim))
                        scheduler.step(temp_val_loss)
                        if temp_val_loss < val_loss:
                            print('val_loss improved from {} to {}, saving model to {}' \
                                  .format(val_loss, temp_val_loss, args.save_model_path + added_path + args.model_name))
                            path_posix = args.save_model_path + added_path + args.model_name
                            save_path = path_posix + '.h5'
                            torch.save(model.state_dict(), save_path)
                            #torch.save(model.state_dict(), save_model_path / model_name)
                            val_loss = temp_val_loss
                        early_stopping(temp_val_loss)
                        if early_stopping.early_stop:
                            break

    else:
        if args.resume or args.fine_tuning:
            if os.path.isfile(resume_path):
                print("=> loading checkpoint '{}'".format(resume_path))
                if device == 'cpu':
                    checkpoint = torch.load(resume_path)
                else:
                    if args.from_ae_to_binary:
                        #to remove the n_out=3 last layer and rename the binaery_layer as head_layer for binary segmentation and use the already binary trained
                        model = load_ae_inference(resume_path=resume_path, device=device, n_features_start=16, n_out=n_out, ae_bin=args.ae_bin,
                                           fine_tuning=args.fine_tuning, unfreezed_layers=args.unfreezed_layers)
                    elif args.from_ae_to_bin:
                        #to replace n_out = 3 with ###new weight#### for a binary layer on top (use to fine tune an autoencoder)
                        model = load_ae_to_bin(resume_path=resume_path, device=device, n_features_start=16, n_out=n_out,
                                           fine_tuning=args.fine_tuning, unfreezed_layers=args.unfreezed_layers)

                    else:
                        model = load_model(resume_path=resume_path, device=device, n_features_start=16, n_out=n_out, ae_bin=args.ae_bin,
                                           fine_tuning=args.fine_tuning, unfreezed_layers=args.unfreezed_layers)

                if args.new_model_name:
                    args.model_name = args.new_model_name
            else:
                print("=> no checkpoint found at '{}'".format(resume_path))
                if args.c0:
                    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out,
                                                        pretrained=False, progress=True)).to(device)
                else:
                    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out, c0=False,
                                                        pretrained=False, progress=True)).to(device)

        else:
            if args.c0:
                model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out, ae_bin=args.ae_bin,
                                              pretrained=False, progress=True,
                                              device=device).to(device))
            else:
                model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out, c0=False, ae_bin=args.ae_bin,
                                              pretrained=False, progress=True,
                                              device=device).to(device))


        val_loss = 10 ** 16
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=args.patience_lr,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=9e-8, verbose=True)
        early_stopping = EarlyStopping(patience=args.patience)
        #Train Loop####
        """
        Set the model to the training mode first and train
        """

        #torch.autograd.set_detect_anomaly(True)
        for epoch in range(args.epochs):
            model.train()
            with tqdm(train_loader, unit="batch") as tepoch:
                for i, (image, target) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    optimizer.zero_grad()

                    y = target.to(device)
                    x = image.to(device)
                    out = model(x)
                    loss = criterion(out, y)
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
                        for i, (image, target) in enumerate(vepoch):
                            optimizer.zero_grad()

                            y = target.to(device)
                            x = image.to(device)
                            out = model(x)
                            loss = criterion(out, y)
                            temp_val_loss += loss
                            if i % 10 == 0:
                                print("VALIDATION Loss: {} batch {} on total of {}".format(loss.item(),
                                                                                        i, len(validation_loader)))

                        temp_val_loss = temp_val_loss / len(validation_loader)
                        print('validation_loss {}'.format(temp_val_loss))
                        scheduler.step(temp_val_loss)
                        if temp_val_loss < val_loss:
                            print('val_loss improved from {} to {}, saving model to {}' \
                                  .format(val_loss, temp_val_loss, args.save_model_path + added_path + args.model_name))
                            print("saving model to {}".format(args.save_model_path + added_path + args.model_name))
                            path_posix = args.save_model_path + added_path + args.model_name
                            save_path = path_posix + '_{}.h5'.format(epoch)
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
    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--save_model_path', nargs="?", default=ModelResults,
                        help='the folder including the masks to crop')
    parser.add_argument('--model_name', nargs="?", default='c-resunet_y',
                        help='model_name')
    parser.add_argument('--new_model_name', nargs="?", default='c-resunet',
                        help='the name the model will have after resume another model name')
    parser.add_argument('--loss_type', nargs="?", default='weighted',
                        help='what kind of loss among weighted, unweightedAE, unweighted')
    parser.add_argument('--patience', nargs="?", type=int, default=5, help='patience for checkpoint')
    parser.add_argument('--patience_lr', nargs="?", type=int, default=3, help='patience for early stopping')
    parser.add_argument('--lr', nargs="?", type=int, default= 0.006, help='learning rate value')
    parser.add_argument('--epochs', nargs="?", type=int, default=200, help='number of epochs')
    parser.add_argument('--bs', nargs="?", type=int, default=8, help='batch size')
    parser.add_argument('--dataset', nargs="?", default='yellow', help='dataset flavour')

    parser.add_argument('--c0', type=int,default=1,
                        help='include or not c0 lauyer')

    parser.add_argument('--ae', action='store_true',
                        help='autoencoder train of resunet')
    parser.add_argument('--ae_bin', action='store_true',
                        help='autoencoder train of resunet with binary layer')
    parser.add_argument('--ae_no_c0', action='store_true',
                        help='autoencoder without c0')

    parser.add_argument('--few_shot', action='store_true',
                        help='use a small dataset to train the model')
    parser.add_argument('--few_shot_merged', action='store_true',
                        help='yellow and red merged')
    parser.add_argument('--self_supervised', action='store_true',
                        help='use labels generated by the model')
    parser.add_argument('--weakly_supervised_ae', action='store_true',
                        help='use labels generated by the model')
    parser.add_argument('--weakly_supervised_ae_bin', action='store_true',
                        help='use labels generated by the model')
    parser.add_argument('--unsupervised', action='store_true',
                        help='use labels generated by the model')

    parser.add_argument('--resume', action='store_true',
                        help='resume training of the model specified with the model name')
    parser.add_argument('--resume_path', default=ModelResults)

    parser.add_argument('--fine_tuning', action='store_true',
                        help='fine tune the model or not')
    parser.add_argument('--from_ae_to_binary', action='store_true',
                        help='fine tune the model coming from autoencoder with pre binary layer')
    parser.add_argument('--from_ae_to_bin', action='store_true',
                        help='fine tune the model coming from autoencoder with pre binary layer')
    parser.add_argument('--unfreezed_layers', default=1,
                        help='number of layer to unfreeze for fine tuning can be a number or a block [encoder, decoder, head]')
    parser.add_argument('--monitor_epochs', action='store_true',
                        help='save each epoch with different model name if val loss is improved')
    args = parser.parse_args()

    if not (Path(args.save_model_path).exists()):
        print('creating path')
        os.makedirs(args.save_model_path)
    train(args=args)


