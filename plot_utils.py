import torch
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn
import seaborn as sns




def get_z(model, data_loader):
        """
        Calculates latent space (z), true RUL and predicted RUL for validation dataset
        :return: 3 np.arrays
        """
        device = "cpu"
        model.eval()
        model.to(device)
        z_space = []
        target = []
        predicted_y = []
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                pairs_mode = data_loader.dataset.return_pairs

                if pairs_mode:
                    x, pos_x, neg_x, true_rul, _, _ = data
                    x, true_rul = x.to(device), true_rul.to(device)
                    predicted_rul, z, mean, log_var, x_hat = model(x)

                else:
                    x, true_rul = data
                    x, true_rul = x.to(device), true_rul.to(device)
                    predicted_rul, z, mean, log_var, x_hat = model(x)

                z_space.append(z.cpu().detach().numpy())
                target.append(true_rul.cpu().detach().numpy().squeeze())
                predicted_y.append(predicted_rul.cpu().detach().numpy().squeeze())
        return np.concatenate(z_space), np.concatenate(target), np.concatenate(predicted_y)


def viz_latent_space(z, y, title='Final', save=True, show=False):
    """
    Plots latent space.
    :param title: Title of the plot, str
    :param save: whether to save the plot or not
    :param show: whether to show the plot or not
    """
    plt.figure(figsize=(8, 4))
    if len(y) > 0:
        plt.scatter(z[:, 0], z[:, 1], c=y, s=1.5)
    else:
        plt.scatter(z[:, 0], z[:, 1])
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    plt.colorbar()
    plt.title(title)
    if save:
        plt.savefig('./images/latent_space_epoch'+str(title)+'.png')
    if show:
        plt.show()



def get_engine_runs(dataloader, model):
    """
    Performs inference for each engine_id (unit number) run from validation dataset
    :return: dictionary with true RUL, predicted RUL and latent spase vector z for each engine_id, dict
    """
    engine_ids = dataloader.dataset.ids
    history = defaultdict(dict)
    model.eval().to('cpu')

    for engine_id in engine_ids:
        engine_id =int(engine_id)
        with torch.no_grad():
            x, y = dataloader.dataset.get_run(engine_id)
            y_hat, z, *_ = model(x)
            history[engine_id]['rul'] = y.numpy()
            history[engine_id]['rul_hat'] = y_hat.numpy()
            history[engine_id]['z'] = z.numpy()

    return history


def plot_engine_run(history, engine_id=None, title="engine_run", save=False):
    engine_ids = history.keys()

    if engine_id is None:
        engine_id = random.choice(list(engine_ids))

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
        plt.savefig("./images/" + str(title) + f"_eng_{engine_id}" + ".png")
    plt.show()


def viz_latent_space_noise(model, dataloader, engine_id=None, mean=0, std=1, n_runs=6, title="FD003_noise", save=True, show=True):
    # TODO: check dataloader outputs! Might be written to fit old dataloader without tripletloss!

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    engine_ids = dataloader.dataset.ids
    if engine_id is None:
        engine_id = random.choice(list(engine_ids))
        
    #Get latent space without noise:
    z_space = []
    targets = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)
            y_hat, z, *_ = model(x)
            z_space.append(z.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    z_space = np.concatenate(z_space)
    targets = np.concatenate(targets)
    targets = np.squeeze(targets)
    
    # Z-space with noise:
    z_space_noise = []
    z_space_noise_y = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)
            x += torch.empty_like(x).normal_(mean=mean,std=std)
            y_hat, z, *_ = model(x)
            z_space_noise.append(z.detach().cpu().numpy()) 
            z_space_noise_y.append(y.detach().cpu().numpy()) 
    z_space_noise = np.concatenate(z_space_noise)
    z_space_noise_y = np.concatenate(z_space_noise_y)
    z_space_noise_y = np.squeeze(z_space_noise_y)

    #Get engine runs with noise:
    engine_z = []
    engine_y_hat = []
    engine_y_true = []
    
    with torch.no_grad():
        x, y = dataloader.dataset.get_run(engine_id)
        x, y = x.to(device), y.to(device)
        y_hat, z, *_ = model(x)
    engine_z_nonoise = z.detach().cpu().numpy()
    engine_y_nonoise = y.detach().cpu().numpy()

    for run in range(n_runs):
        with torch.no_grad():
            x, y = dataloader.dataset.get_run(engine_id)
            x, y = x.to(device), y.to(device)
            x += torch.empty_like(x).normal_(mean=mean,std=std)
            y_hat, z, *_ = model(x)
            engine_z.append(z.detach().cpu().numpy())
            engine_y_true.append(y.detach().cpu().numpy())
    engine_z = np.concatenate(engine_z)
    engine_y_true = np.concatenate(engine_y_true)
    engine_y_true = np.squeeze(engine_y_true)
    engine_y_true = np.squeeze(engine_y_true)
    
    
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12, 10))
    targets = np.squeeze(targets)
    
   
    pc = ax[0][0].scatter(z_space[:, 0], z_space[:, 1], c=targets, s=5, alpha=1)
    divider = make_axes_locatable(ax[0][0])
    ccx = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(pc, cax=ccx)
    ax[0][0].set_title("Latent Space w/o noise")
    ax[0][0].set_xlabel("z - dim 1")
    ax[0][0].set_ylabel("z - dim 2")
    
    pd = ax[1][0].scatter(z_space_noise[:, 0], z_space_noise[:, 1], c=z_space_noise_y, s=5, alpha=1)
    divider = make_axes_locatable(ax[1][0])
    cdx = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(pd, cax=cdx)
    ax[1][0].set_title(f"Latent Space with noise $\mu$={mean} $\sigma$={std}")
    ax[1][0].set_xlabel("z - dim 1")
    ax[1][0].set_ylabel("z - dim 2")
    
    pa = ax[0][1].scatter(z_space[:, 0], z_space[:, 1], c=targets, s=5, alpha=1)
    pf=ax[0][1].scatter(engine_z[:, 0], engine_z[:, 1],c=engine_y_true, s=5, alpha=1, cmap=plt.cm.gist_heat_r)
    divider = make_axes_locatable(ax[0][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cfx = divider.append_axes("right", size="5%", pad=0.6)
    plt.colorbar(pf, cax=cfx)
    plt.colorbar(pa, cax=cax)
    ax[0][1].set_title(f"{n_runs} trajectories of Engine Unit #{engine_id} with noise.")
    ax[0][1].set_xlabel("z - dim 1")
    ax[0][1].set_ylabel("z - dim 2")

    
    engine_y_true = np.squeeze(engine_y_true)
    #"YlGnBu_r" "YlOrBr_r"
    divider = make_axes_locatable(ax[1][1])
    cbx = divider.append_axes("right", size="5%", pad=0.05)
    cbbx = divider.append_axes("right", size="5%", pad=0.6)
    #sns.kdeplot(x=engine_z[:, 0], y=engine_z[:, 1], ax=ax[1][1], cmap="Reds",  fill=True, thresh=0.01, alpha=1.0, cbar=True, cbar_ax = cbx, cut=1)
    pb = ax[1][1].scatter(z_space[:, 0], z_space[:, 1], c=targets, s=5, alpha=1)
    sns.kdeplot(x=engine_z[:, 0], y=engine_z[:, 1], ax=ax[1][1], cmap="coolwarm", fill=True, thresh=0.02, alpha=0.7, cbar=True, cbar_ax = cbbx, cut=3)
    plt.colorbar(pb, cax=cbx)
    ax[1][1].scatter(engine_z_nonoise[:, 0], engine_z_nonoise[:, 1], c=engine_y_nonoise, s=6, cmap=plt.cm.gist_heat_r, alpha=1)
    ax[1][1].set_title(f"PDF of {n_runs} trajectories of Engine Unit #{engine_id} with noise.")
    ax[1][1].set_xlabel("z - dim 1")
    ax[1][1].set_ylabel("z - dim 2")    
    
    if show:
        plt.tight_layout()
    if save:
        plt.savefig("./images/" + str(title) + f"_std_{std}" + f"_eng_{engine_id}" + ".png")
    plt.show()


def get_test_score(model, data_loader):
    """
    Calculates score and RMSE on test dataset
    :return: score and RMSE, int
    """
    def score_fn(y, y_hat):
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
    device = "cpu"
    rmse = 0
    score = 0
    model.eval()
    model.to(device)
    pairs_mode = data_loader.dataset.return_pairs
    for batch_idx, data in enumerate(data_loader):
        with torch.no_grad():
            if pairs_mode:
                x, pos_x, neg_x, true_rul, _, _ = data
            else:
                x, true_rul = data
            x, true_rul = x.to(device), true_rul.to(device)

            predicted_rul, *_ = model(x)

            loss = nn.MSELoss()(predicted_rul, true_rul)

            rmse += loss.item() * len(true_rul)
            score += score_fn(true_rul, predicted_rul).item()

    rmse = (rmse / len(data_loader.dataset)) ** 0.5
    print(f"RMSE: {rmse :6.3f} Score: {score :6.3f}")
    return score, rmse