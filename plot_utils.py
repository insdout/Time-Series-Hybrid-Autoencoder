import torch
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn
import seaborn as sns
import os 
import pandas as pd




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


def get_engine_runs_diffusion(dataloader, rve_model, diffusion_model, device='cpu'):
    """
    Performs inference for each engine_id (unit number) run from validation dataset
    :return: dictionary with true RUL, predicted RUL and latent spase vector z for each engine_id, dict
    """
    engine_ids = dataloader.dataset.ids
    history = defaultdict(dict)
    rve_model.eval().to(device)
    diffusion_model.eval().to(device)

    for engine_id in engine_ids:
        engine_id =int(engine_id)
        with torch.no_grad():
            x, y = dataloader.dataset.get_run(engine_id)
            x = x.to(device)
            y = y.to(device)
            y_hat, z, *_ = rve_model(x)
            print("z shape:", z.shape, "x size:", x.shape)
            x_hat, _ = diffusion_model.sample_cmapss(n_sample=1, size=(1,32,32), device=x.device, z_space_contexts=z, guide_w = 0.0)
            
            #TODO: get several samples and pick the best one
            #================================================
            
            print("shape x_hat", x_hat.shape)
            rul_hat_diff, z_diff, *_ = rve_model(x_hat[:,:,:,:21].squeeze(1))

            history[engine_id]['rul'] = y.detach().cpu().numpy()
            history[engine_id]['rul_hat'] = y_hat.detach().cpu().numpy()
            history[engine_id]['z'] = z.detach().cpu().numpy()
            history[engine_id]["x"] = x.detach().cpu().numpy()
            history[engine_id]["x_hat"] = x_hat[:,:,:,:21].squeeze(1).detach().cpu().numpy()
            history[engine_id]["rul_hat_diff"] = rul_hat_diff.detach().cpu().numpy()
            history[engine_id]["z_diff"] = z_diff.detach().cpu().numpy()

    return history


def plot_engine_run_diff(
    history, 
    img_path="./outputs/diffusion_outputs/images_decision/", 
    engine_id=None, 
    title="engine_run", 
    save=False
    ):
    """_summary_

    Args:
        history (_type_): _description_
        img_path (str, optional): _description_. Defaults to "./outputs/diffusion_outputs/images_decision/".
        engine_id (_type_, optional): _description_. Defaults to None.
        title (str, optional): _description_. Defaults to "engine_run".
        save (bool, optional): _description_. Defaults to False.
    """    
    engine_ids = history.keys()

    if engine_id is None:
        engine_id = random.choice(list(engine_ids))

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(12, 6))
    real_rul = history[engine_id]['rul']
    rul_hat = history[engine_id]['rul_hat']
    rul_hat_diff =  history[engine_id]['rul_hat_diff']
    ax[0].plot(real_rul)
    ax[0].plot(rul_hat)
    ax[0].plot(rul_hat_diff, color="darkviolet")
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
    pb = ax[1].scatter(z[:, 0], z[:, 1], c=targets, s=10, cmap=plt.cm.gist_heat_r)
    cbb = plt.colorbar(pb, shrink=1.0)
    cbb.set_label(f"Engine #{engine_id} RUL")
    ax[1].set_xlabel("z - dim 1")
    ax[1].set_ylabel("z - dim 2")
    
    z_diff = history[engine_id]['z_diff']
    targets = history[engine_id]['rul']
    pb2 = ax[1].scatter(z_diff[:, 0], z_diff[:, 1], c=targets, s=10, cmap=plt.cm.cool_r, alpha=1)
    cbb2 = plt.colorbar(pb2, shrink=1.0)
    cbb2.set_label(f"Engine #{engine_id} RUL by diffusion")
    ax[1].set_xlabel("z - dim 1")
    ax[1].set_ylabel("z - dim 2")
    
    if save:
        #img_path ="./outputs/diffusion_outputs/images/" 
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path + str(title) + f"_eng_{engine_id}" + ".png")
    plt.show()


def plot_engine_run_diff_decision_boundary(
    rve_model, history, 
    img_path="./outputs/diffusion_outputs/images_decision/", 
    engine_id=None, title="engine_run", 
    save=False
    ):
    """_summary_

    Args:
        rve_model (_type_): _description_
        history (_type_): _description_
        img_path (str, optional): _description_. Defaults to "./outputs/diffusion_outputs/images_decision/".
        engine_id (_type_, optional): _description_. Defaults to None.
        title (str, optional): _description_. Defaults to "engine_run".
        save (bool, optional): _description_. Defaults to False.
    """    
    engine_ids = history.keys()

    if engine_id is None:
        engine_id = random.choice(list(engine_ids))

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(12, 6))
    real_rul = history[engine_id]['rul']
    rul_hat = history[engine_id]['rul_hat']
    rul_hat_diff =  history[engine_id]['rul_hat_diff']
    ax[0].plot(real_rul)
    ax[0].plot(rul_hat)
    ax[0].plot(rul_hat_diff, color="darkviolet")
    ax[0].set_title(f"Engine Unit #{engine_id}")
    ax[0].set_xlabel("Time(Cycle)")
    ax[0].set_ylabel("RUL")

    with torch.no_grad():
        z1_lim = [-1.5, 4]
        z2_lim = [-2, 6]

        h = 0.01
        rve_model.eval()
        rve_model.cpu()
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(z1_lim[0], z1_lim[1], h), np.arange(z2_lim[0], z2_lim[1], h))
      
        # Predict the function value for the whole gid
        z_mesh = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).unsqueeze(0)
        rul_hat = rve_model.regressor(z_mesh)
        rul_hat = rul_hat.reshape(xx.shape)
        # Plot the contour and training examples
        levels = list(range(125))
        pb = ax[1].contourf(xx, yy, rul_hat, levels=levels)
        cbb = plt.colorbar(pb, shrink=1.0)
        cbb.set_label("RVE Regressor RUL decision")
        
        z_diff = history[engine_id]['z_diff']
        targets = history[engine_id]['rul']
        pb2 = ax[1].scatter(z_diff[:, 0], z_diff[:, 1], c=targets, s=5, cmap=plt.cm.cool, alpha=1)
        cbb2 = plt.colorbar(pb2, shrink=1.0)
        cbb2.set_label(f"Engine #{engine_id} RUL by diffusion")
        ax[1].set_xlabel("z - dim 1")
        ax[1].set_ylabel("z - dim 2")
    
    if save:
        #img_path ="./outputs/diffusion_outputs/images_decision/" 
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path + str(title) + f"_eng_{engine_id}" + ".png")
    plt.show()
    
    
def reconstruct_timeseries(history, engine_id, rul_delta_threshold=60):
    """
    Reconstructs Time-Series from array of data slices (window_size, n_sensors) by appending last row of each consequent window,
    to thw first one. Data slices with RUL delta above RUL delta threshold are taken as a whole, overwriting the constructed Time-Series,
    as we need to highlight the problems in generative process if they exist.

    Args:
        history (_type_): _description_
        engine_id (_type_): _description_
        rul_delta_threshold (int, optional): _description_. Defaults to 60.

    Returns:
        _type_: _description_
    """    
    rul_true = history[engine_id]["rul"]
    rul_predicted = history[engine_id]["rul_hat"]
    x =  history[engine_id]["x"]
    z = history[engine_id]["z"] 
    x_diff = history[engine_id]["x_hat"] 
    rul_hat_diff = history[engine_id]["rul_hat_diff"].squeeze(1)

    rul_delta = np.abs(rul_true - rul_hat_diff).astype(int)
    rul_delta_mask = rul_delta > rul_delta_threshold
    x_reconstructed = []
    x_diff_reconstructed = []
    """
    print("rul delta mask", rul_delta_mask)
    print("rul shape", rul_true.shape)
    print("rul_hat_diff shape", rul_hat_diff.shape)
    print("rul delta",rul_delta.shape, rul_delta)
    """
    for ind, rul in enumerate(rul_true):
        if ind == 0:
            #print("first shape:", x_diff[ind].shape)
            x_diff_reconstructed.append(x_diff[ind])
            x_reconstructed.append(x[ind])
        else:
            #print("second shape:", x_diff[ind, -1, :].shape)
            x_diff_reconstructed.append(np.expand_dims(x_diff[ind, -1, :], axis=0))
            x_reconstructed.append(np.expand_dims(x[ind, -1, :], axis=0))
    

    rul_delta_mask_indexes = rul_delta_mask.nonzero()[0]
    print(f"engine_id: {engine_id} rul_delta_diff index size: {rul_delta_mask_indexes.shape[0]}")
    x_diff_reconstructed = np.concatenate(x_diff_reconstructed, axis=0)
    x_reconstructed = np.concatenate(x_reconstructed, axis=0)
    
    for delta_indx in rul_delta_mask_indexes:
        #print(delta_indx, type(delta_indx))
        x_diff_reconstructed[0 + delta_indx: 32 + delta_indx] = x_diff[delta_indx]
    #print("output shapes", x_diff_reconstructed.shape, x_reconstructed.shape)
    return x_reconstructed, x_diff_reconstructed, rul_true, rul_hat_diff, rul_predicted


def plot_sensors(df_true, df_diff,  rul_true, rul_hat_diff, rul_predicted, engine_id, path, save, show):
    """
    Plots sensor data for both: the original input data X from the dataset and reconstructed X_hat from diffusion model.

    Args:
        df_true (_type_): _description_
        df_diff (_type_): _description_
        rul_true (_type_): _description_
        rul_hat_diff (_type_): _description_
        rul_predicted (_type_): _description_
        engine_id (_type_): _description_
        path (_type_): _description_
        save (_type_): _description_
        show (_type_): _description_
    """    
    timesteps, num_sensors = df_true.shape
    fig , ax = plt.subplots(num_sensors+1, 1, sharex=False, figsize=(8, 3*num_sensors))

    ax[0].plot(rul_true, color="green", label="true RUL")
    ax[0].plot(rul_hat_diff, color="red", label="diffusion RUL")
    ax[0].plot(rul_predicted, color="orange", label="RVE RUL")
    ax[0].set_title("RUL vs Time")
    ax[0].legend(loc = "upper right")
    
    for i, sensor in enumerate(df_true.columns):
        i +=1
        ax[i].plot(df_true[sensor], color="green", label="true X")
        ax[i].plot(df_diff[sensor], color="red", label="diffusion X_hat")
        ax[i].set_title(f"sensor: {sensor}")
        ax[i].legend(loc = "upper right")
    
    fig.suptitle(f"Engine_id: {engine_id}")
    if show:
        fig.tight_layout()
        fig.subplots_adjust(top=0.97)
        
    if save:
        fig.subplots_adjust(top=0.97)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path + "sensors" + f"_eng_{engine_id}" + ".png", dpi=50)
        
    
def reconstruct_and_plot(history, engine_id, path, save=True, show=True):
    """
    Reconstructs Time-Series from history dict converts them into pandas DataFrame 
    and plots all sensors (both original and generated)

    Args:
        history (_type_): _description_
        engine_id (_type_): _description_
        path (_type_): _description_
        save (bool, optional): _description_. Defaults to True.
        show (bool, optional): _description_. Defaults to True.
    """    
    x_true, x_diff, rul_true, rul_hat_diff, rul_predicted = reconstruct_timeseries(history, engine_id=engine_id)
    columns = [f"s_{ind}" for ind in range(x_true.shape[1])]

    df_true = pd.DataFrame(x_true, columns=columns)
    df_diff = pd.DataFrame(x_diff, columns=columns)
    
    plot_sensors(df_true, df_diff,  rul_true, rul_hat_diff, rul_predicted, engine_id=engine_id, path=path, save=save, show=show)