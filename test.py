import os
import os.path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from metric import KNNRULmetric
import json
import gc

import hydra


class Tester:

    @staticmethod
    def score(true_rul, rul_hat):
        """
        Computes score according to original CMAPSS dataset paper.
        :param y: true RUL
        :param y_hat: predicted RUL
        :return: float
        """
        score = 0
        true_rul = true_rul.cpu()
        rul_hat = rul_hat.cpu()
        for i in range(len(rul_hat)):
            if true_rul[i] <= rul_hat[i]:
                score += np.exp(-(true_rul[i] - rul_hat[i]) / 10.0) - 1
            else:
                score += np.exp((true_rul[i] - rul_hat[i]) / 13.0) - 1
        return score
    
    def __init__(self, 
                 path, 
                 model, 
                 val_loader, 
                 test_loader, 
                 rul_threshold=125, 
                 n_neighbors=3, 
                 add_noise_val=False, 
                 add_noise_test=False, 
                 noise_mean=0, 
                 noise_std=1, 
                 save=True, 
                 show=False
                 ):
        self.model = model
        self.metric = KNNRULmetric(rul_threshold=rul_threshold, n_neighbors=n_neighbors)
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = "cpu"
        self.add_noise_val = add_noise_val
        self.add_noise_test = add_noise_test
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        z, true_rul, rul_hat = self.get_z()
        self.z = z
        self.true_rul = true_rul
        self.rul_hat = rul_hat
        self.path = path
        self.engine_history = self.get_engine_runs()
        self.save = save
        self.show = show

    def get_test_score(self):
        """
        Calculates score and RMSE on test dataset
        :return: score and RMSE, int
        """
        rmse = 0
        score = 0
        self.model.eval()
        for batch_idx, data in enumerate(self.test_loader):
            with torch.no_grad():
                x, true_rul = data
                # Adding Gaussian noise if add_noise_test == True:
                if self.add_noise_test:
                    x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                x, true_rul = x.to(self.device), true_rul.to(self.device)

                rul_hat, *_ = self.model(x)
                
                loss = nn.MSELoss()(rul_hat, true_rul)

                rmse += loss.item() * len(true_rul)
                score += Tester.score(true_rul, rul_hat).item()

        rmse = (rmse / len(self.test_loader.dataset)) ** 0.5
        #print(f"RMSE: {rmse :6.3f} Score: {score :6.3f}")
        return score, rmse

    def get_z(self):
        """
        Calculates latent space (z), true RUL and predicted RUL for validation dataset
        :return: 3 np.arrays
        """
        self.model.eval()
        self.model.to('cpu')
        z_space = []
        true_rul = []
        predicted_rul = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                batch_len = len(self.val_loader.dataset)
                pairs_mode = self.val_loader.dataset.return_pairs

                if pairs_mode:
                    x, pos_x, neg_x, y, _, _ = data
                    # Adding Gaussian noise if add_noise_val == True:
                    if self.add_noise_val:
                        x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat, z, mean, log_var, x_hat = self.model(x)

                else:
                    x, y = data
                    # Adding Gaussian noise if add_noise_val == True:
                    if self.add_noise_val:
                        x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat, z, mean, log_var, x_hat = self.model(x)
                   
                z_space.append(z.numpy())
                true_rul.append(y.numpy())
                predicted_rul.append(y_hat.numpy())
        return np.concatenate(z_space), np.concatenate(true_rul), np.concatenate(predicted_rul)

    def viz_latent_space(self, title='Final', save=True, show=True):
        """
        Plots latent space.
        :param title: Title of the plot, str
        :param save: whether to save the plot or not
        :param show: whether to show the plot or not
        """
        z = self.z
        targets = self.true_rul
        plt.figure(figsize=(8, 4))
        if len(targets) > 0:
            plt.scatter(z[:, 0], z[:, 1], c=targets, s=1.5)
        else:
            plt.scatter(z[:, 0], z[:, 1])
        plt.xlabel('z - dim 1')
        plt.ylabel('z - dim 2')
        plt.colorbar()
        plt.title(title)
        if save:
            images_dir  =  os.path.join(self.path, "images")
            os.makedirs(images_dir, exist_ok=True)
            file_name = 'latent_space_epoch' + str(title) + '.png'
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

    def get_engine_runs(self):
        """
        Performs inference for each engine_id (unit number) run from validation dataset
        :return: dictionary with true RUL, predicted RUL and latent spase vector z for each engine_id, dict
        """
        engine_ids = self.val_loader.dataset.ids
        history = defaultdict(dict)
        self.model.eval().to('cpu')

        for engine_id in engine_ids:
            with torch.no_grad():
                x, true_rul = self.val_loader.dataset.get_run(engine_id)
                # Adding Gaussian noise if add_noise_val == True:
                if self.add_noise_val:
                    x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                rul_hat, z, *_ = self.model(x)
                history[engine_id]['rul'] = true_rul.numpy()
                history[engine_id]['rul_hat'] = rul_hat.numpy()
                history[engine_id]['z'] = z.numpy()

        return history
    
    def plot_engine_run(self, title="engine_run", save=True, show=False):
        """
        Plots each engine_id (unit number) trajectory over whole latent space of validation dataset.
        :param title: Title of the plot, str
        :param save: whether to save the plot or not
        :param show: whether to show the plot or not
        """
        history = self.engine_history
        engine_ids = history.keys()

        for engine_id in engine_ids:

            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(12, 6))
            true_rul = history[engine_id]['rul']
            rul_hat = history[engine_id]['rul_hat']
            ax[0].plot(true_rul)
            ax[0].plot(rul_hat)
            ax[0].set_title(f"Engine Unit #{engine_id}")
            ax[0].set_xlabel("Time(Cycle)")
            ax[0].set_ylabel("RUL")
            ax[0].set_yticks(list(range(0, 130, 25)))
            ax[0].grid(True)
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
                images_dir  =  os.path.join(self.path, "images")
                os.makedirs(images_dir, exist_ok=True)
                file_name = str(title) + f"_eng_{engine_id}" + ".png"
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
            #else:
                #plt.close(fig)

    def test(self):
        """
        Calls latent space visualization function and engine_run plot function.
        """
        self.viz_latent_space(save=self.save, show=self.show)
        self.plot_engine_run(save=self.save, show=self.show)
        # Calculate score and rmse on test dataset:
        score, rmse = self.get_test_score()
        # Calculate latent space metric for validation dataset:
        metric = self.metric.fit_calculate(z=self.z, rul=self.true_rul.ravel())
        self.metric.plot_zspace(
            z=self.z, 
            rul=self.true_rul.ravel(), 
            path=self.path, 
            title=str(round(metric, 4)), 
            save=self.save, 
            show=self.show
            )
        results = {"test_score": score, "test_rmse": rmse, "val_metric": metric}
        print(f"TEST Score: {score :6.4f}, RMSE: {rmse :6.4f}")
        with self.safe_open_w(self.path +"/results.json") as f:
            json.dump(results, f)

    def safe_open_w(self, path):
        ''' 
        Open "path" for writing, creating any parent directories as needed.
        :param path:
        :return: object
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
    z, t, true_rul = tester.get_z()
    print(z.shape, true_rul.shape, true_rul.shape, len(val_loader.dataset))
    print(tester.get_test_score())
    tester.test()


if __name__ == "__main__":
    main()
