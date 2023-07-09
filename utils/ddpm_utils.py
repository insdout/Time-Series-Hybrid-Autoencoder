from collections import defaultdict
import torch
from tslearn.metrics import dtw, lcss, gak 
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np


def load_latent_trajectory(trajectory_file_path):
    """
    Load the latent trajectory from a JSON file.
    
    Args:
        trajectory_file_path (str): Path to the JSON file containing the latent trajectory.
    
    Returns:
        dict: Dictionary containing the loaded latent trajectory, where each key corresponds to an engine ID
            and the value is a NumPy array of shape (n_samples, 2) representing the latent coordinates.
    """ 
    with open(trajectory_file_path) as fp:
        val_latent = json.load(fp)
        
    for key in val_latent.keys():
        val_latent[key] = np.stack([val_latent[key][0], val_latent[key][1]]).T
    
    return val_latent


def get_diffusion_outputs_from_dataloader(
        dataloader, 
        tshae_model, 
        diffusion_model, 
        engine_id=None,
        num_samples=1, 
        w=0.0, 
        quantile=0.25, 
        mode='quantile', 
        device='cuda'
        ):
    """
    Performs inference for each engine_id (unit number) run from validation dataset
    :return: dictionary with true RUL, predicted RUL and latent spase vector z for each engine_id, dict
    """
    from collections import defaultdict
    
    if engine_id:
        engine_ids = [engine_id]
    else:
        engine_ids = dataloader.dataset.ids
    history = defaultdict(dict)
    tshae_model.eval().to(device)
    diffusion_model.eval().to(device)

    for engine_id in engine_ids:
        engine_id =int(engine_id)
        with torch.no_grad():
            x, y = dataloader.dataset.get_run(engine_id)
            x = x.to(device)
            y = y.to(device)

            #print(f" xshape: {x.shape} y shape: {y.shape}")
            y_hat, z, *_, x_tshae_samples = tshae_model(x)

            num_z = z.shape[0]
            x_hat_diffusion, _ = diffusion_model.sample_cmapss(n_sample=num_samples, size=(1,32,32), device=x.device, z_space_contexts=z, guide_w = w)
            
            #TODO: get several samples and pick the best one
            #================================================
            z = z.unsqueeze(1)

            x_hat_diffusion = x_hat_diffusion.squeeze(1)[:,:,:21]
            with torch.no_grad(): 
                rul_hat_diff, z_diff, *_  = tshae_model(x_hat_diffusion)
            z_diff = z_diff.reshape(num_samples,num_z, 2).permute(1,0,2)
 
            rul_hat_diff = rul_hat_diff.reshape(num_samples, num_z,  1).permute(1,0,2)

            distances = torch.cdist(z, z_diff)

            if mode == "quantile":
          
                limits = torch.quantile(distances.squeeze(1), quantile, interpolation='linear', dim=1, keepdim=True)
                choices = []
                for i in range(distances.shape[0]):
                    choices.append(np.random.choice(np.flatnonzero(np.where(distances.squeeze(1).detach().cpu().numpy()[i] < limits.detach().cpu().numpy()[i], 1, 0)), 1))
                
                best_samples = torch.from_numpy(np.array(choices).squeeze(1))
            else:
                best_samples = torch.argmin(distances.squeeze(1), dim=1)
            
            x_diff_samples = x_hat_diffusion.reshape(num_samples, num_z, 32, 21).permute(1,0,2,3)

            x_diff_samples = x_diff_samples[range(x_diff_samples.shape[0]), best_samples]
            z_diff = z_diff[range(z_diff.shape[0]), best_samples]
            rul_hat_diff = rul_hat_diff[range(rul_hat_diff.shape[0]), best_samples]
            
            x_diff_samples = x_diff_samples.to("cpu")
            x_tshae_samples = x_tshae_samples.to("cpu")
            
            sensors_diff_reconstructed = []
            sensors_tshae_reconstructed = []
            sensors_true_reconstructed = []
            x_true, _ = dataloader.dataset.get_run(int(engine_id))
            x_true_samples = x_true[-num_z:, :, :]
            for ind in range(num_z):
                if ind == 0:
                    #print("first shape:", x_diff[ind].shape)
                    sensors_diff_reconstructed.append(x_diff_samples[ind])
                    sensors_tshae_reconstructed.append(x_tshae_samples[ind])
                    sensors_true_reconstructed.append(x_true_samples[ind])
                else:
                    #print("second shape:", x_diff[ind, -1, :].shape)
                    sensors_diff_reconstructed.append(np.expand_dims(x_diff_samples[ind, -1, :], axis=0))
                    sensors_tshae_reconstructed.append(np.expand_dims(x_tshae_samples[ind, -1, :], axis=0))
                    sensors_true_reconstructed.append(np.expand_dims(x_true_samples[ind, -1, :], axis=0))

            sensors_true_reconstructed_full = []
            for ind in range(x_true.shape[0]):
                if ind == 0:
                    sensors_true_reconstructed_full.append(x_true[ind])
                else:
                    sensors_true_reconstructed_full.append(np.expand_dims(x_true[ind, -1, :], axis=0))

            sensors_diff_reconstructed = np.concatenate(sensors_diff_reconstructed, axis=0)
            sensors_tshae_reconstructed = np.concatenate(sensors_tshae_reconstructed, axis=0)
            sensors_true_reconstructed = np.concatenate(sensors_true_reconstructed, axis=0)
            sensors_true_reconstructed_full = np.concatenate(sensors_true_reconstructed_full, axis=0)


            history[engine_id]['z'] = z.squeeze().detach().cpu().numpy()
            history[engine_id]["z_diff"] = z_diff.detach().cpu().numpy()
            
            history[engine_id]["x_true_samples"] = x_true_samples
            history[engine_id]["x_tshae_samples"] = x_tshae_samples.detach().cpu().numpy()
            history[engine_id]["x_diff_samples"] = x_diff_samples.detach().cpu().numpy()
            
            
            history[engine_id]['rul'] = y.detach().cpu().numpy()
            history[engine_id]['rul_hat'] = y_hat.detach().cpu().numpy()
            history[engine_id]["rul_hat_diff"] = rul_hat_diff.detach().cpu().numpy()
            
            
            history[engine_id]["sensors_diff_reconstructed"] =  sensors_diff_reconstructed
            history[engine_id]["sensors_tshae_reconstructed"] =  sensors_tshae_reconstructed
            history[engine_id]["sensors_true_reconstructed"] =  sensors_true_reconstructed
            history[engine_id]["sensors_true_reconstructed_full"] =  sensors_true_reconstructed_full
            
    return history


def get_diffusion_outputs_from_z(
        z_space_dict, 
        tshae_model, 
        diffusion_model,
        dataloader, 
        engine_id=None,
        num_samples=4, 
        w=0.5, 
        quantile=0.25, 
        mode='best', 
        device='cuda'
        ):
    """
    Performs inference for each engine_id (unit number) run from validation dataset
    :return: dictionary with true RUL, predicted RUL and latent spase vector z for each engine_id, dict
    """
    from collections import defaultdict
    
    if engine_id:
        engine_ids = [str(engine_id)]
    else:
        engine_ids = z_space_dict.keys()

    history = defaultdict(dict)
    tshae_model.eval().to(device)
    diffusion_model.eval().to(device)

    for engine_id in engine_ids:
        with torch.no_grad():
            z = z_space_dict[engine_id]
            z = torch.FloatTensor(z).to(device)
            #print(f" xshape: {x.shape} y shape: {y.shape}")
            #y_hat, z, *_ = tshae_model(x)
            print(f"z shape: {z.shape}")
            x_tshae_samples = tshae_model.decoder(z)
            num_z = z.shape[0]
            x_hat_diffusion, _ = diffusion_model.sample_cmapss(n_sample=num_samples, size=(1,32,32), device=z.device, z_space_contexts=z, guide_w = w)
            
            #TODO: get several samples and pick the best one
            #================================================
            z = z.unsqueeze(1)

            x_hat_diffusion = x_hat_diffusion.squeeze(1)[:,:,:21]
            with torch.no_grad(): 
                rul_hat_diff, z_diff, *_  = tshae_model(x_hat_diffusion)
            z_diff = z_diff.reshape(num_samples,num_z, 2).permute(1,0,2)
 
            rul_hat_diff = rul_hat_diff.reshape(num_samples, num_z,  1).permute(1,0,2)

            distances = torch.cdist(z, z_diff)

            if mode == "quantile":
          
                limits = torch.quantile(distances.squeeze(1), quantile, interpolation='linear', dim=1, keepdim=True)
                choices = []
                for i in range(distances.shape[0]):
                    choices.append(np.random.choice(np.flatnonzero(np.where(distances.squeeze(1).detach().cpu().numpy()[i] < limits.detach().cpu().numpy()[i], 1, 0)), 1))
                
                best_samples = torch.from_numpy(np.array(choices).squeeze(1))
            else:
                best_samples = torch.argmin(distances.squeeze(1), dim=1)
            
            x_diff_samples = x_hat_diffusion.reshape(num_samples, num_z, 32, 21).permute(1,0,2,3)

            x_diff_samples = x_diff_samples[range(x_diff_samples.shape[0]), best_samples]
            z_diff = z_diff[range(z_diff.shape[0]), best_samples]
            rul_hat_diff = rul_hat_diff[range(rul_hat_diff.shape[0]), best_samples]
            
            x_diff_samples = x_diff_samples.to("cpu")
            x_tshae_samples = x_tshae_samples.to("cpu")
            
            sensors_diff_reconstructed = []
            sensors_tshae_reconstructed = []
            sensors_true_reconstructed = []
            x_true, _ = dataloader.dataset.get_run(int(engine_id))
            x_true_samples = x_true[-num_z:, :, :]
            for ind in range(num_z):
                if ind == 0:
                    #print("first shape:", x_diff[ind].shape)
                    sensors_diff_reconstructed.append(x_diff_samples[ind])
                    sensors_tshae_reconstructed.append(x_tshae_samples[ind])
                    sensors_true_reconstructed.append(x_true_samples[ind])
                else:
                    #print("second shape:", x_diff[ind, -1, :].shape)
                    sensors_diff_reconstructed.append(np.expand_dims(x_diff_samples[ind, -1, :], axis=0))
                    sensors_tshae_reconstructed.append(np.expand_dims(x_tshae_samples[ind, -1, :], axis=0))
                    sensors_true_reconstructed.append(np.expand_dims(x_true_samples[ind, -1, :], axis=0))

            sensors_true_reconstructed_full = []
            for ind in range(x_true.shape[0]):
                if ind == 0:
                    sensors_true_reconstructed_full.append(x_true[ind])
                else:
                    sensors_true_reconstructed_full.append(np.expand_dims(x_true[ind, -1, :], axis=0))

            sensors_diff_reconstructed = np.concatenate(sensors_diff_reconstructed, axis=0)
            sensors_tshae_reconstructed = np.concatenate(sensors_tshae_reconstructed, axis=0)
            sensors_true_reconstructed = np.concatenate(sensors_true_reconstructed, axis=0)
            sensors_true_reconstructed_full = np.concatenate(sensors_true_reconstructed_full, axis=0)


            history[engine_id]['z'] = z.squeeze().detach().cpu().numpy()
            history[engine_id]["z_diff"] = z_diff.detach().cpu().numpy()
            
            history[engine_id]["x_true_samples"] = x_true_samples
            history[engine_id]["x_tshae_samples"] = x_tshae_samples.detach().cpu().numpy()
            history[engine_id]["x_diff_samples"] = x_diff_samples.detach().cpu().numpy()
           
            
            history[engine_id]["rul_hat_diff"] = rul_hat_diff.detach().cpu().numpy()
            
            history[engine_id]["sensors_diff_reconstructed"] =  sensors_diff_reconstructed
            history[engine_id]["sensors_tshae_reconstructed"] =  sensors_tshae_reconstructed
            history[engine_id]["sensors_true_reconstructed"] =  sensors_true_reconstructed
            history[engine_id]["sensors_true_reconstructed_full"] =  sensors_true_reconstructed_full

    return history


def plot_results_reconstruction(history, path):
    """
    Plot and save the results of sensor signal reconstruction.

    Args:
        history (dict): Dictionary containing the sensor signal reconstruction history.
            The keys represent the engine IDs, and the values are dictionaries containing the following keys:
            - "sensors_true_reconstructed": True sensor signals.
            - "sensors_diff_reconstructed": Diffusion-based reconstructed sensor signals.
            - "sensors_tshae_reconstructed": Decoder-based reconstructed sensor signals.
            - "x_true_samples": True sensor signal samples.
            - "x_diff_samples": Diffusion-based reconstructed sensor signal samples.
            - "x_tshae_samples": Decoder-based reconstructed sensor signal samples.
            - "sensors_true_reconstructed_full": Full true sensor signals (including the future samples).

        path (str): Path to the directory where the results will be saved.

    Returns:
        None
    """
    data = defaultdict(list)
    engine_ids = list(history.keys())
    for engine_id in engine_ids:
        data["engine_id"].append(engine_id)
        real = history[engine_id]["sensors_true_reconstructed"]
        predicted_diff = history[engine_id]["sensors_diff_reconstructed"]
        predicted_rve = history[engine_id]["sensors_tshae_reconstructed"]
        data["dtw diffusion"].append(dtw(real, predicted_diff))
        data["lcss diffusion"].append(lcss(real, predicted_diff))
        data["rmse diffusion"].append(mean_squared_error(real, predicted_diff))
        data["dtw rve"].append(dtw(real, predicted_rve))
        data["lcss rve"].append(lcss(real, predicted_rve))
        data["rmse rve"].append(mean_squared_error(real, predicted_rve))
    df = pd.DataFrame(data)
    print(df.mean())
    
    reconstruction_dir  =  os.path.join(path, "sensor_reconstruction")
    os.makedirs(reconstruction_dir, exist_ok=True)
    file_name = "metric.csv"
    csv_path = os.path.join(reconstruction_dir, file_name)
    df.to_csv(csv_path, index=False)
    
    
    for engine_id in engine_ids:
        x_true = history[engine_id]["x_true_samples"]
        x_diff = history[engine_id]["x_diff_samples"]
        x_rve = history[engine_id]["x_tshae_samples"]
        
        n_samples = x_true.shape[0]
        frames = range(0, n_samples, 10)
        fig, ax = plt.subplots(nrows=3, ncols=len(frames), sharex=True, sharey=True, figsize=(2*len(frames), 8))
        
        for ax_index, i in enumerate(frames):
            ax[0][ax_index].imshow(x_true[i])
            ax[0][ax_index].set_title(f"True at RUL: {45-i}", fontsize=12)
            ax[0][ax_index].set_xticks([])
            ax[0][ax_index].set_yticks([])
            ax[1][ax_index].imshow(x_diff[i])
            ax[1][ax_index].set_title(f"Diffusion at RUL: {45-i}", fontsize=12)
            ax[1][ax_index].set_xticks([])
            ax[1][ax_index].set_yticks([])
            ax[2][ax_index].imshow(x_rve[i])
            ax[2][ax_index].set_title(f"Decoder at RUL: {45-i}", fontsize=12)
            ax[2][ax_index].set_xticks([])
            ax[2][ax_index].set_yticks([])
        plt.suptitle(f'Sensor Signals for Unit Number {engine_id}', fontsize=16)
        
 
        images_dir  =  os.path.join(reconstruction_dir, "frames")
        os.makedirs(images_dir, exist_ok=True)
        file_name =  "frames" + f"_eng_{engine_id}" + ".png"
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, file_name))
        print(f"Saved image at {os.path.join(images_dir, file_name)}")
        plt.cla()
        plt.clf()
        plt.close('all')

        sensors_diff = history[engine_id]["sensors_diff_reconstructed"]
        sensors_true = history[engine_id]["sensors_true_reconstructed_full"]
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        x = list(range(sensors_true.shape[0]))
        x_last_reconstructed = x[-sensors_diff.shape[0]:]
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10, 7))
        ax1 = plt.subplot2grid((3,5), (0,0), colspan=3)
        ax2 = plt.subplot2grid((3,5), (1,0), colspan=3)
        ax3 = plt.subplot2grid((3,5), (2, 0), colspan=3)
        ax4 = plt.subplot2grid((3,5), (0, 3), rowspan=3)
        ax5 = plt.subplot2grid((3,5), (0, 4), rowspan=3)
        ax1.plot(x, sensors_true[:, 6], color=colors[1], label="True")
        ax1.plot(x_last_reconstructed, sensors_diff[:, 6], color="darkviolet", label="Diffusion")
        ax1.legend(loc="upper left", fontsize=12)
        ax1.set_title(f"Sensor s_7", fontsize=14)
        ax1.tick_params(axis="both", which="major", labelsize=14)
        ax1.tick_params(axis="both", which="minor", labelsize=14)        
        
        ax2.plot(x, sensors_true[:, 8], color=colors[1], label="True")
        ax2.plot(x_last_reconstructed, sensors_diff[:, 8], color="darkviolet", label="Diffusion")
        ax2.legend(loc="upper left", fontsize=12)
        ax2.set_title(f"Sensor s_9", fontsize=14)
        ax2.tick_params(axis="both", which="major", labelsize=14)
        ax2.tick_params(axis="both", which="minor", labelsize=14) 
        
        ax3.plot(x, sensors_true[:, 13], color=colors[1], label="True")
        ax3.plot(x_last_reconstructed, sensors_diff[:, 13], color="darkviolet", label="Diffusion")
        ax3.legend(loc="upper left", fontsize=12)
        ax3.set_title(f"Sensor s_14", fontsize=14)
        ax3.tick_params(axis="both", which="major", labelsize=14)
        ax3.tick_params(axis="both", which="minor", labelsize=14) 
        
        ax4.imshow(sensors_true[-sensors_diff.shape[0]:])
        ax4.set_title("True", fontsize=14)
        ax4.set_xticks([])
        ax4.set_yticks([])
        
        ax5.imshow(sensors_diff)
        ax5.set_title(f"Diffusion", fontsize=14)
        ax5.set_xticks([])
        ax5.set_yticks([])        
        plt.suptitle(f'Sensor Signals for Unit Number {engine_id}', fontsize=16)
        
        images_dir  =  os.path.join(reconstruction_dir, "timeseries")
        os.makedirs(images_dir, exist_ok=True)
        file_name =  "frames" + f"_eng_{engine_id}" + ".png"
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, file_name))
        print(f"Saved image at {os.path.join(images_dir, file_name)}")
        plt.cla()
        plt.clf()
        plt.close('all')