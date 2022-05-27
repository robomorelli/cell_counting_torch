from tqdm import tqdm
from model.utils import *

def train_class(model, device, train_loader, validation_loader,
                config):

    epochs = config['epochs']
    lr = config['lr']
    w0 = config['w0']
    w1 = config['w1']

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    """
    Set the model to the training mode first and train
    """
    val_loss = 10 ** 16
    patience = 5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  factor=0.8, patience=patience,
                                            threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=9e-8, verbose=True)

    Wloss = WeightedLoss(w0, w1)
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):

            optimizer.zero_grad()
            x = model(data[0].to(device))
            y = data[1].to(device)
            loss = Wloss(x, y)
            loss.backward()
            optimizer.step()
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

            val_loss_cpu = val_loss.cpu().item()
            print('validation_loss {}'.format(val_loss_cpu))
            scheduler.step(val_loss)
            return val_loss_cpu