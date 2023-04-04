from metric_dataloader import MetricDataPreprocessor
from loss import TotalLoss
import torch
from collections import defaultdict
import json
import logging
import os
import matplotlib.pyplot as plt

import hydra 
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


class Trainer():
    def __init__(self, model, optimizer, train_loader, val_loader, n_epochs, total_loss, validate, save_model, save_history, verbose=True, device=None):
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.total_loss = total_loss
        self.history = defaultdict(list)
        self.verbose = verbose
        self.validate = validate
        self.save_model = save_model
        self.save_history = save_history
        self.log = logging.getLogger(__name__)

    def train_epoch(self):
        epoch_loss = defaultdict(list)
        self.model.train()
        for batch_idx, data in enumerate(self.train_loader):
            batch_len = len(self.train_loader.dataset)
            pairs_mode = self.train_loader.dataset.return_pairs
            self.optimizer.zero_grad()

            if pairs_mode:
                x, pos_x, neg_x, y, _, _ = data
                x = x.to(self.device)
                y = y.to(self.device)
                pos_x = pos_x.to(self.device)
                neg_x = neg_x.to(self.device)

                y_hat, z, mean, log_var, x_hat = self.model(x)
                _, z_pos, *_ = self.model(pos_x)
                _, z_neg, *_ = self.model(neg_x)
                loss_dict = self.total_loss(mean=mean, log_var=log_var, y=y, y_hat=y_hat, x=x, x_hat=x_hat, z=z, z_pos=z_pos, z_neg=z_neg)
            else:
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                y_hat, z, mean, log_var, x_hat = self.model(x)
                loss_dict = self.total_loss(mean=mean, log_var=log_var, y=y, y_hat=y_hat, x=x, x_hat=x_hat, z=z)
            
            loss = loss_dict["TotalLoss"]
            loss.backward()
            self.optimizer.step()

            for key in loss_dict:
                epoch_loss[key].append(loss_dict[key].item()*len(y))
        for key in loss_dict:
            self.history["Train_"+key].append(sum(epoch_loss[key])/batch_len)


    def valid_epoch(self):
        epoch_loss = defaultdict(list)
        self.model.train()
        for batch_idx, data in enumerate(self.val_loader):
            batch_len = len(self.val_loader.dataset)
            pairs_mode = self.val_loader.dataset.return_pairs
            with torch.no_grad():

                if pairs_mode:
                    x, pos_x, neg_x, y, _, _ = data
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pos_x = pos_x.to(self.device)
                    neg_x = neg_x.to(self.device)

                    y_hat, z, mean, log_var, x_hat = self.model(x)
                    _, z_pos, *_ = self.model(pos_x)
                    _, z_neg, *_ = self.model(neg_x)

                    loss_dict = self.total_loss(mean=mean, log_var=log_var, y=y, y_hat=y_hat, x=x, x_hat=x_hat, z=z, z_pos=z_pos, z_neg=z_neg)
                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat, z, mean, log_var, x_hat = self.model(x)
                    loss_dict = self.total_loss(mean=mean, log_var=log_var, y=y, y_hat=y_hat, x=x, x_hat=x_hat, z=z)
        

                for key in loss_dict:
                    epoch_loss[key].append(loss_dict[key].item()*len(y))
        for key in loss_dict:
            self.history["Val_"+key].append(sum(epoch_loss[key])/batch_len)

    def train(self, n_epochs=None):
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
        if n_epochs:
            self.n_epochs = n_epochs
        for epoch in range(self.n_epochs):
            self.train_epoch()
            if self.validate:
                self.valid_epoch()
            if self.verbose:
                for key in self.history:
                    self.log.info(f"Epoch:{epoch} {key}: {self.history[key][-1] :3.3f}")
        if self.save_model:
            torch.save(self.model, output_dir+"/rve_model.pt")
            print("Saved", output_dir+"/rve_model.pt")
        if self.save_history:
            with open(output_dir+"/history.json", 'w') as fp:
                json.dump(self.history, fp)
        self.plot_learning_curves(output_dir)
    
    def plot_learning_curves(self, path, show=True, save=True):
        history = self.history
        fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(16, 6))

        ax[0][0].plot(history['Train_TotalLoss'], label='train loss')
        ax[0][0].plot(history['Val_TotalLoss'], label='val loss')

        ax[0][1].plot(history['Train_KLLoss'], label='train kl_loss loss')
        ax[0][1].plot(history['Val_KLLoss'], label='val kl_loss loss')

        ax[1][0].plot(history['Train_RegLoss'], label='train reg loss')
        ax[1][0].plot(history['Val_RegLoss'], label='val reg loss')

        ax[1][1].plot(history['Train_ReconLoss'], label='train reconstruction loss')
        ax[1][1].plot(history['Val_ReconLoss'], label='val reconstruction loss')

        ax[2][0].plot(history['Train_TripletLoss'], label='train triplet loss')
        ax[2][0].plot(history['Val_TripletLoss'], label='val triplet loss')

        ax[0][0].legend(loc='upper right')
        ax[0][1].legend(loc='upper right')
        ax[1][0].legend(loc='upper right')
        ax[1][1].legend(loc='upper right')
        ax[2][0].legend(loc='upper right')
        #ax[2][1].legend(loc='upper right')
        ax[0][0].grid(True)
        ax[0][1].grid(True)
        ax[1][0].grid(True)
        ax[1][1].grid(True)
        ax[2][0].grid(True)
        #ax[2][1].grid(True)
        ax[0][0].set_yscale('log')
        ax[0][1].set_yscale('log')
        ax[1][0].set_yscale('log')
        ax[1][1].set_yscale('log')
        ax[2][0].set_yscale('log')
        #ax[2][1].set_yscale('log')
        if save:
            img_path = path+'/images/'
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.savefig(img_path + "Training Curves" + ".png")
        if show:
            plt.show()






@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):

    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    encoder = instantiate(config.model.encoder)
    decoder = instantiate(config.model.decoder)
    model_rve = instantiate(config.model.rve, encoder=encoder, decoder=decoder)

    optimizer = instantiate(config.optimizer,  params=model_rve.parameters())

    total_loss = TotalLoss(config)

    trainer = Trainer(**config.trainer,
        model=model_rve, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        total_loss=total_loss, 
        )
    trainer.train()

if __name__ == "__main__":
    main()