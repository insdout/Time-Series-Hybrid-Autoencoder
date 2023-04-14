import os, os.path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
import random

import hydra 
from omegaconf import DictConfig, OmegaConf

"""""
with safe_open_w('/Users/bill/output/output-text.txt') as f:
    f.write(...)
"""

class Tester():

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
    
    def __init__(self, path, model, val_loader, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = "cpu"
        z, y, y_hat = self.get_z()
        self.z = z
        self.y = y
        self.y_hat = y_hat
        self.path = path
        self.engine_history = self.get_engine_runs()


    def get_test_score(self):
        rmse = 0
        score = 0
        self.model.eval()
        for batch_idx, data in enumerate(self.test_loader):
            with torch.no_grad():
                x, y = data
                x, y = x.to(self.device), y.to(self.device)

                y_hat, *_ = self.model(x)
                
                loss = nn.MSELoss()(y_hat, y)

                rmse += loss.item() * len(y)
                score += Tester.score(y, y_hat).item()

        rmse = (rmse / len(self.test_loader.dataset)) ** 0.5
        print(f"RMSE: {rmse :6.3f} Score: {score :6.3f}")
        return score, rmse

    def get_z(self):
        self.model.eval()
        self.model.to('cpu')
        z_space = []
        target = []
        predicted_y = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                batch_len = len(self.val_loader.dataset)
                pairs_mode = self.val_loader.dataset.return_pairs

                if pairs_mode:
                    x, pos_x, neg_x, y, _, _ = data

                    y_hat, z, mean, log_var, x_hat = self.model(x)
                   

                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat, z, mean, log_var, x_hat = self.model(x)
                   
                z_space.append(z.numpy())
                target.append(y.numpy())
                predicted_y.append(y_hat.numpy())
        return np.concatenate(z_space), np.concatenate(target), np.concatenate(predicted_y)

    def viz_latent_space(self, title='Final', save=True, show=True):
        z = self.z
        targets = self.y
        plt.figure(figsize=(8, 4))
        if len(targets)>0:
            plt.scatter(z[:, 0], z[:, 1], c=targets, s=1.5)
        else:
            plt.scatter(z[:, 0], z[:, 1])
        plt.xlabel('z - dim 1')
        plt.ylabel('z - dim 2')
        plt.colorbar()
        plt.title(title)
        if save:
            img_path = self.path+'/images/'
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.savefig(img_path +'latent_space_epoch'+str(title)+'.png')
        if show:
            plt.show()

    def get_engine_runs(self):
        engine_ids = self.val_loader.dataset.ids
        history = defaultdict(dict)
        self.model.eval().to('cpu')

        for engine_id in engine_ids:
            with torch.no_grad():
                x, y = self.val_loader.dataset.get_run(engine_id)
                y_hat, z, *_ = self.model(x)
                history[engine_id]['rul'] = y.numpy()
                history[engine_id]['rul_hat'] = y_hat.numpy()
                history[engine_id]['z'] = z.numpy()

        return history
    
    def plot_engine_run(self, title="engine_run", save=True, show=False):
        history = self.engine_history
        engine_ids = history.keys()

        for engine_id in engine_ids:

            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(12, 6))
            real_rul = history[engine_id]['rul']
            rul_hat = history[engine_id]['rul_hat']
            ax[0].plot(real_rul)
            ax[0].plot(rul_hat)
            ax[0].set_title(f"Engine Unit #{engine_id}")
            ax[0].set_xlabel("Time(Cycle)")
            ax[0].set_ylabel("RUL")
            for run in engine_ids:
                z = history[run]['z']
                targets = history[run]['rul']
                pa = ax[1].scatter(z[:, 0], z[:, 1], c=targets, s=1.5)
            cba = plt.colorbar(pa, shrink=1.0)
            cba.set_label("RUL")

            z = history[engine_id]['z']
            targets = history[engine_id]['rul']
            pb = ax[1].scatter(z[:, 0], z[:, 1], c=targets, s=15, cmap=plt.cm.gist_heat_r)
            cbb = plt.colorbar(pb, shrink=1.0)
            cbb.set_label(f"Engine #{engine_id} RUL")
            ax[1].set_xlabel("z - dim 1")
            ax[1].set_ylabel("z - dim 2")

            if save:
                img_path = self.path+'/images/'
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                plt.savefig(img_path + str(title) + f"_eng_{engine_id}" + ".png")
            if show:
                plt.show()

    def test(self):
        self.viz_latent_space()
        self.plot_engine_run()

    def safe_open_w(self, path):
        ''' 
        Open "path" for writing, creating any parent directories as needed.
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, 'w')


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    from metric_dataloader import MetricDataPreprocessor
    path = "/home/mikhail/Thesis/MDS-Thesis-RULPrediction/outputs/2023-04-12/00-37-45/"
    model = torch.load(path+"rve_model.pt")

    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    tester = Tester(path, model, val_loader, test_loader)
    z, t, y = tester.get_z()
    print(z.shape, y.shape, y.shape, len(val_loader.dataset))
    print(tester.get_test_score())
    tester.test()

   

if __name__ == "__main__":
    main()