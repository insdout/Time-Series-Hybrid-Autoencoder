from utils.metric_dataloader import MetricDataPreprocessor
from utils.loss import TotalLoss
import torch
from collections import defaultdict
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import gc

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class Trainer:

    @staticmethod
    def score(y, y_hat):
        """
        Computes score according to original CMAPSS dataset paper.
        :param y: true RUL
        :param y_hat: predicted RUL
        :return: float
        """
        score = 0
        y = y.cpu()
        y_hat = y_hat.cpu()
        for i in range(len(y_hat)):
            if y[i] <= y_hat[i]:
                score += np.exp(-(y[i] - y_hat[i]) / 10.0) - 1
            else:
                score += np.exp((y[i] - y_hat[i]) / 13.0) - 1
        return score

    def __init__(self, model, optimizer, train_loader, val_loader, test_loader, n_epochs, total_loss, validate,
                 save_model, save_history, add_noise_train=False, add_noise_val=False, noise_mean=0, noise_std=1, scheduler=None, verbose=True, device=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        self.add_noise_train = add_noise_train
        self.add_noise_val = add_noise_val
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.log = logging.getLogger(__name__)

    def train_epoch(self):
        """
        Training loop. Performs iteration over dataset, computing loss functions and updating the model`s weights.
        """
        epoch_loss = defaultdict(list)
        self.model.train()
        for batch_idx, data in enumerate(self.train_loader):
            batch_len = len(self.train_loader.dataset)
            pairs_mode = self.train_loader.dataset.return_pairs
            self.optimizer.zero_grad()

            # If Triplet Loss will be calculated, the dataset should return 3 datapoints:
            # anchor, positive and negative:
            if pairs_mode:
                x, pos_x, neg_x, true_rul, _, _ = data
                if self.add_noise_train:
                        x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                        pos_x += torch.empty_like(pos_x).normal_(mean=self.noise_mean, std=self.noise_std)
                        neg_x += torch.empty_like(neg_x).normal_(mean=self.noise_mean, std=self.noise_std)
                x = x.to(self.device)
                true_rul = true_rul.to(self.device)
                pos_x = pos_x.to(self.device)
                neg_x = neg_x.to(self.device)

                predicted_rul, z, mean, log_var, x_hat = self.model(x)
                _, z_pos, *_ = self.model(pos_x)
                _, z_neg, *_ = self.model(neg_x)
                # Computing the total loss:
                loss_dict = self.total_loss(
                    mean=mean,
                    log_var=log_var,
                    y=true_rul,
                    y_hat=predicted_rul,
                    x=x,
                    x_hat=x_hat,
                    z=z,
                    z_pos=z_pos,
                    z_neg=z_neg
                )
            # If no Triplet loss will be used, the dataset should return 1 datapoint:
            else:
                x, true_rul = data
                if self.add_noise_train:
                        x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                x, true_rul = x.to(self.device), true_rul.to(self.device)
                predicted_rul, z, mean, log_var, x_hat = self.model(x)
                # Computing the total loss:
                loss_dict = self.total_loss(
                    mean=mean,
                    log_var=log_var,
                    y=true_rul,
                    y_hat=predicted_rul,
                    x=x,
                    x_hat=x_hat,
                    z=z
                )

            loss = loss_dict["TotalLoss"]
            loss.backward()
            self.optimizer.step()

            # Updating the history dictionary:
            for key in loss_dict:
                epoch_loss[key].append(loss_dict[key].item() * len(true_rul))
        
        # Updating learning rate if scheduler != None:
        if self.scheduler:
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]
                self.log.info(f"learning_rate: {lr}")
        # Updating history dictionary:
        for key in loss_dict:
            self.history["Train_" + key].append(sum(epoch_loss[key]) / batch_len)

    def valid_epoch(self):
        """
        Validation loop. Performs iteration over dataset, computing loss functions and updating the history dictionary.
        """
        epoch_loss = defaultdict(list)
        self.model.train()
        for batch_idx, data in enumerate(self.val_loader):
            batch_len = len(self.val_loader.dataset)
            pairs_mode = self.val_loader.dataset.return_pairs
            with torch.no_grad():

                if pairs_mode:
                    x, pos_x, neg_x, y, _, _ = data
                    if self.add_noise_val:
                        x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                        pos_x += torch.empty_like(pos_x).normal_(mean=self.noise_mean, std=self.noise_std)
                        neg_x += torch.empty_like(neg_x).normal_(mean=self.noise_mean, std=self.noise_std)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pos_x = pos_x.to(self.device)
                    neg_x = neg_x.to(self.device)

                    y_hat, z, mean, log_var, x_hat = self.model(x)
                    _, z_pos, *_ = self.model(pos_x)
                    _, z_neg, *_ = self.model(neg_x)

                    loss_dict = self.total_loss(mean=mean, log_var=log_var, y=y, y_hat=y_hat, x=x, x_hat=x_hat, z=z,
                                                z_pos=z_pos, z_neg=z_neg)
                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    if self.add_noise_val:
                        x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                    y_hat, z, mean, log_var, x_hat = self.model(x)
                    loss_dict = self.total_loss(mean=mean, log_var=log_var, y=y, y_hat=y_hat, x=x, x_hat=x_hat, z=z)

                for key in loss_dict:
                    epoch_loss[key].append(loss_dict[key].item() * len(y))
        for key in loss_dict:
            self.history["Val_" + key].append(sum(epoch_loss[key]) / batch_len)

    def train(self, n_epochs=None):
        """
        Calls train and validate epochs functions, calculates the RMSE and score values,
        optionally saves the model on the last epoch
        :param n_epochs: number of training epochs
        """
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
        best_rmse = float("inf")
        if n_epochs:
            self.n_epochs = n_epochs
        for epoch in range(self.n_epochs):
            self.train_epoch()
            if self.validate:
                self.valid_epoch()
            if self.verbose:
                for key in self.history:
                    self.log.info(f"Epoch:{epoch} {key}: {self.history[key][-1] :3.3f}")
            score, rmse = self.get_dataset_score(self.val_loader)
            self.history["Validation_score"].append(score)
            self.history["Validation_RMSE"].append(rmse)
            writer.add_scalar("Score/Test", score, epoch)
            writer.add_scalar("RMSE/Test", rmse, epoch)
            print(f"rmse:{rmse :6.3f} best rmse: {best_rmse :6.3f}")
            if rmse < best_rmse:
                best_rmse = rmse
                if self.save_model:
                    torch.save(self.model, output_dir + "/best_rve_model.pt")
                    print(f"Saved best model, rmse: {rmse :6.3f} at {output_dir}/best_rve_model.pt")

        if self.save_model:
            torch.save(self.model, output_dir + "/rve_model.pt")
            print("Saved", output_dir + "/rve_model.pt")
        if self.save_history:
            with open(output_dir + "/history.json", 'w') as fp:
                json.dump(self.history, fp)
        self.plot_learning_curves(output_dir, show=False)
        writer.flush()
        writer.close()

    def get_dataset_score(self, datasetloader):
        """
        Calculates score and RMSE on test dataset
        :return: score and RMSE, int
        """
        rmse = 0
        score = 0
        self.model.eval()
        pairs_mode = datasetloader.dataset.return_pairs
        for batch_idx, data in enumerate(datasetloader):
            with torch.no_grad():
                if pairs_mode:
                    x, _, _, y, _, _ = data
                else:
                    x, y = data
                x, y = x.to(self.device), y.to(self.device)

                y_hat, *_ = self.model(x)

                loss = nn.MSELoss()(y_hat, y)

                rmse += loss.item() * len(y)
                score += Trainer.score(y, y_hat).item()

        rmse = (rmse / len(self.test_loader.dataset)) ** 0.5
        print(f"RMSE: {rmse :6.3f} Score: {score :6.3f}")
        return score, rmse

    def plot_learning_curves(self, path, show=True, save=True):
        """
        Plots learning curves from history dict.
        :param path: path to the directory where plots will be saved, string
        :param show: whether display the plots or not, Bool
        :param save: whether save the plots or not, Bool
        """
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

        ax[2][1].plot(history['Validation_score'], label='Valisation Score')
        ax2 = ax[2][1].twinx()
        ax2.plot(history['Validation_RMSE'], color='orange', label='Val RMSE')
        ax[2][1].set_ylabel('Validation Score', color='blue')
        ax2.set_ylabel('Validation RMSE', color='orange')
        lines, labels = ax[2][1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        ax[0][0].legend(loc='upper right')
        ax[0][1].legend(loc='upper right')
        ax[1][0].legend(loc='upper right')
        ax[1][1].legend(loc='upper right')
        ax[2][0].legend(loc='upper right')
        # ax[2][1].legend(loc='upper right')
        ax[0][0].grid(True)
        ax[0][1].grid(True)
        ax[1][0].grid(True)
        ax[1][1].grid(True)
        ax[2][0].grid(True)
        ax[2][1].grid(True)
        ax[0][0].set_yscale('log')
        ax[0][1].set_yscale('log')
        ax[1][0].set_yscale('log')
        ax[1][1].set_yscale('log')
        ax[2][0].set_yscale('log')
        # ax[2][1].set_yscale('log')
        if save:
            images_dir  =  os.path.join(path, "images")
            os.makedirs(images_dir, exist_ok=True)
            file_name =  "Training Curves.png"
            plt.tight_layout()
            plt.savefig(os.path.join(images_dir, file_name))
            # Clear the current axes.
            plt.cla() 
            # Clear the current figure.
            plt.clf() 
            # Closes all the figure windows.
            plt.close('all')   
            gc.collect()
        if show:
            plt.show()


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    from tshae_test import Tester

    # fix random seeds:
    if config.random_seed.fix == True:
        import random
        import os

        torch.manual_seed(config.random_seed.seed)
        torch.cuda.manual_seed_all(config.random_seed.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True) 
        np.random.seed(config.random_seed.seed)
        random.seed(config.random_seed.seed)
        # see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


    # current hydra output folder
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    # Preparing dataloaders:
    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    # Instantiating the model:
    encoder = instantiate(config.model.encoder)
    decoder = instantiate(config.model.decoder)
    model_rve = instantiate(config.model.rve, encoder=encoder, decoder=decoder)

    # Instantiating the optimizer:
    optimizer = instantiate(config.optimizer, params=model_rve.parameters())

    # Instantiating the learning rate scheduler:
    if config.scheduler.mode.enable:
        scheduler = instantiate(config.scheduler.self, optimizer=optimizer)
    else:
        scheduler = None

    # Instantiating the TotalLoss class:
    total_loss = TotalLoss(config)

    # Initializing the trainer:
    trainer = Trainer(**config.trainer.trainer,
                      model=model_rve,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      total_loss=total_loss,
                      scheduler=scheduler
                      )
    trainer.train()

    # Loading trained and saved model from previous step:
    path = output_dir
    model = torch.load(path + "/best_rve_model.pt")

    # Running test utils:
    rul_threshold = config.knnmetric.rul_threshold
    n_neighbors = config.knnmetric.n_neighbors
    tester = Tester(
        **config.trainer.tester, 
        path=path, 
        model=model, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        rul_threshold=rul_threshold, 
        n_neighbors=n_neighbors
        )
    tester.test()


if __name__ == "__main__":
    main()