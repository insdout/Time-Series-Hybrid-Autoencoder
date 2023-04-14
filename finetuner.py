from metric_dataloader import MetricDataPreprocessor
from loss import FineTuneTotalLoss
import torch
from collections import defaultdict
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from metric import knnRULmetric
import torch.nn as nn

import math

import hydra 
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


class FineTuner():

    @staticmethod
    def score(y, y_hat):
        score = 0
        y = y.cpu()
        y_hat = y_hat.cpu()
        for i in range(len(y_hat)):
            if y[i] <= y_hat[i]:
                score += np.exp(-(y[i] - y_hat[i]) / 10.0) - 1
            else:
                score += np.exp((y[i] - y_hat[i]) / 13.0) - 1
        return score
    
    def __init__(self, 
                 model, 
                 optimizer, 
                 train_loader, 
                 val_loader, 
                 test_loader,
                 n_epochs, 
                 total_loss, 
                 validate, 
                 save_model, 
                 save_history, 
                 metric_rul_threshold,
                 metric_n_neighbors,
                 verbose=True, 
                 device=None):
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
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
        self.metric = knnRULmetric(rul_threshold=metric_rul_threshold, n_neighbors=metric_n_neighbors)

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
            print(key, sum(epoch_loss[key])/batch_len, batch_len)
            if math.isnan(sum(epoch_loss[key])/batch_len):
                print("ERROR!")
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
            print()
            print("Epoch:", epoch)
            self.train_epoch()
            if self.validate:
                self.valid_epoch()
            
            # Inside epoch actions:
            z_train, y_train, y_hat_tarin = self.get_z(self.train_loader)
            train_score, train_rmse = self.get_test_score(self.train_loader)
            self.history["Train_score"].append(train_score)
            self.history["Train_rmse"].append(train_rmse)
            self.viz_latent_space(z_train, y_train, title=f"train_epoch_{epoch}", save=True, show=False)
            train_metric = self.metric.fit_calculate(z_train, y_train)
            self.history["Train_metric"].append(train_metric)
            self.metric.plot_zspace(z_train, y_train, save=True, show=False, path=output_dir, title=f"train_epoch_{epoch}_metric_{train_metric :3.4f}")

            z_val, y_val, y_hat_val = self.get_z(self.val_loader)
            val_score, val_rmse = self.get_test_score(self.val_loader)
            self.history["Val_score"].append(val_score)
            self.history["Val_rmse"].append(val_rmse)
            self.viz_latent_space(z_val, y_val, title=f"val_epoch_{epoch}", save=True, show=False)
            val_metric = self.metric.fit_calculate(z_val, y_val)
            self.history["Val_metric"].append(val_metric)
            self.metric.plot_zspace(z_val, y_val, save=True, show=False, path=output_dir, title=f"val_epoch_{epoch}_metric_{val_metric :3.4f}")
            
            z_test, y_test, y_hat_test = self.get_z(self.test_loader)
            test_score, test_rmse = self.get_test_score(self.test_loader)
            self.history["Test_score"].append(test_score)
            self.history["Test_rmse"].append(test_rmse)
            self.viz_latent_space(z_test, y_test, title=f"test_epoch_{epoch}", save=True, show=False)
            test_metric = self.metric.fit_calculate(z_test, y_test)
            self.history["test_metric"].append(test_metric)
            self.metric.plot_zspace(z_test, y_test, save=True, show=False, path=output_dir, title=f"test_epoch_{epoch}_metric_{test_metric :3.4f}")
            # Add tensorboard scores and metric
            writer.add_scalar("Tune/Score/Test", test_score, epoch)
            writer.add_scalar("Tune/RMSE/Test", test_rmse, epoch)
            writer.add_scalar("Tune/Metric/Test", test_metric, epoch)

            if self.verbose:
                for key in self.history:
                    self.log.info(f"Epoch:{epoch} {key}: {self.history[key][-1] :3.3f}")

        if self.save_model:
            torch.save(self.model, output_dir+"/rve_model_tuned.pt")
            print("Saved", output_dir+"/rve_model.pt")
        if self.save_history:
            with open(output_dir+"/history.json", 'w') as fp:
                json.dump(self.history, fp)
        self.plot_learning_curves(output_dir)
        writer.flush()
        writer.close()
    
    def plot_learning_curves(self, path, show=False, save=True):
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
            plt.close('all')
        if show:
            plt.show()
            
    
    def get_test_score(self, data_loader):
        rmse = 0
        score = 0
        self.model.eval()
        self.model.to(self.device)
        pairs_mode = data_loader.dataset.return_pairs
        for batch_idx, data in enumerate(data_loader):
            with torch.no_grad():
                if pairs_mode:
                    x, pos_x, neg_x, y, _, _ = data
                else:
                    x, y = data
                x, y = x.to(self.device), y.to(self.device)

                y_hat, *_ = self.model(x)
                
                loss = nn.MSELoss()(y_hat, y)

                rmse += loss.item() * len(y)
                score += FineTuner.score(y, y_hat).item()

        rmse = (rmse / len(data_loader.dataset)) ** 0.5
        print(f"RMSE: {rmse :6.3f} Score: {score :6.3f}")
        return score, rmse

    def get_z(self, data_loader):
        self.model.eval()
        self.model.to(self.device)
        z_space = []
        target = []
        predicted_y = []
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                batch_len = len(data_loader.dataset)
                pairs_mode = data_loader.dataset.return_pairs

                if pairs_mode:
                    x, pos_x, neg_x, y, _, _ = data
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat, z, mean, log_var, x_hat = self.model(x)
                   

                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat, z, mean, log_var, x_hat = self.model(x)
                   
                z_space.append(z.cpu().detach().numpy())
                target.append(y.cpu().detach().numpy().squeeze())
                predicted_y.append(y_hat.cpu().detach().numpy().squeeze())
        return np.concatenate(z_space), np.concatenate(target), np.concatenate(predicted_y)

    def viz_latent_space(self, z, y, title='Final', save=True, show=False):
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
        plt.figure(figsize=(8, 4))
        if len(y)>0:
            plt.scatter(z[:, 0], z[:, 1], c=y, s=1.5)
        else:
            plt.scatter(z[:, 0], z[:, 1])
        plt.xlabel('z - dim 1')
        plt.ylabel('z - dim 2')
        plt.colorbar()
        plt.title(title)
        if save:
            img_path = output_dir+'/images/'
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.savefig(img_path +'latent_space_'+str(title)+'.png')
            plt.close('all')
        if show:
            plt.show()
            






@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):

    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

 
    model_rve = torch.load(config.finetuner.checkpoint.path)

    optimizer = instantiate(config.finetuner.optimizer,  params=model_rve.parameters())

    total_loss = FineTuneTotalLoss(config.finetuner)

    tuner = FineTuner(**config.finetuner.tuner,
        model=model_rve, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        total_loss=total_loss, 
        )
    tuner.train()

if __name__ == "__main__":
    main()