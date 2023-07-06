from utils.metric_dataloader import MetricDataPreprocessor
from models.ddpm_models import DDPM, ContextUnet
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import random
import torch
from collections import defaultdict
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate



@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def train_cmapss(config):

    preproc = MetricDataPreprocessor(**config.diffusion.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()
    print(f"train set: {len(train_loader.dataset)} val set: {len(val_loader.dataset)}")

    model_tshae = torch.load(config.diffusion.checkpoint_tshae.path)
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    print(f"output dir: {output_dir}")

  
    n_epoch = config.diffusion.ddpm_train.epochs
    n_T = config.diffusion.ddpm_train.n_T # 500
    device = config.diffusion.ddpm_train.device #"cuda:0" or "cpu"
    z_dim   = config.diffusion.ddpm_train.z_dim
    n_feat = config.diffusion.ddpm_train.n_feat # 128 ok, 256 better (but slower)
    lrate = config.diffusion.ddpm_train.lrate #1e-4
    save_model = config.diffusion.ddpm_train.save_model
    save_dir = output_dir #'./outputs/diffusion_outputs/'
    ws_test = config.diffusion.ddpm_train.ws_test #[0.0, 0.5, 2.0]  strength of generative guidance
    
    drop_prob = config.diffusion.ddpm_model.drop_prob
    
    ddpm = DDPM(
        nn_model=ContextUnet(
        in_channels=1, 
        n_feat=n_feat, 
        z_dim=z_dim), 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=drop_prob)

    ddpm.to(device)
    
    model_tshae.eval().to(device)
    for param in model_tshae.parameters():
        param.requires_grad = False

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    # Instantiating the optimizer:
    optimizer = instantiate(config.diffusion.optimizer, params=ddpm.parameters())

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optimizer.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(train_loader)
        loss_ema = None
        pairs_mode = train_loader.dataset.return_pairs
        for data in pbar:
            if pairs_mode:
                x, pos_x, neg_x, true_rul, _, _ = data
            else:
                x, true_rul = data
            
            x = x.to(device)
            with torch.no_grad():
                predicted_rul, z, mean, log_var, x_hat = model_tshae(x)
                m = nn.ReplicationPad2d((0, 11, 0, 0))
                x_diffusion = m(x)

            optimizer.zero_grad()
            x_diffusion = x_diffusion.unsqueeze(1).to(device)
            context = z.to(device)
            loss = ddpm(x_diffusion, context)
            
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optimizer.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        rul_range = np.arange(0, 100, 10)
        run_ids = train_loader.dataset.ids
        idx = random.choice(run_ids)
        run_x, run_rul = train_loader.dataset.get_run(idx)
        x_samples = run_x[torch.isin(run_rul, torch.Tensor(rul_range))]
        rul_seed = run_rul[torch.isin(run_rul, torch.Tensor(rul_range))]
        x_samples = x_samples.to(device)
        with torch.no_grad():
            predicted_rul, z_samples, mean, log_var, x_hat = model_tshae(x_samples)
                
        
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4 
            num_columns = z_samples.shape[0]
            num_rows = n_sample
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store =ddpm.sample_cmapss(n_sample=n_sample, size=(1,32,32), device=device, z_space_contexts=z_samples, guide_w = w)

                # append some real images at bottom, order by class also
                x_real = m(x_samples).to(device)

                x_all = torch.cat([x_gen, x_real.unsqueeze(1)])

                fig, axs = plt.subplots(nrows=num_rows+1, ncols=num_columns ,sharex=True,sharey=True,figsize=(20,15))
                for row in range(num_rows+1):
                    if row == num_rows:
                        plot_type = "true"
                    else:
                        plot_type = "gen"
                    for col in range(num_columns):
                        axs[row, col].clear()
                        axs[row, col].set_xticks([])
                        axs[row, col].set_yticks([])
                        axs[row, col].set_title(f"{plot_type} Id: {idx} RUL: {int(rul_seed[col])}",  fontsize=10)
                        axs[row, col].imshow(x_all[row*num_columns+col,:,:,:21].cpu().squeeze(),vmin=(x_all[:,:,:,:21].min()), vmax=(x_all[:,:,:,:21].max()))
                img_path = save_dir + '/images/'
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                plt.savefig(img_path + f"image_ep{ep}_w{w}.png", dpi=100)
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")
                plt.close('all')
                #fig.clf()

                if ep%5==0 or ep == int(n_epoch-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns ,sharex=True,sharey=True,figsize=(12,7))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(num_rows):
                            for col in range(num_columns):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                plots.append(axs[row, col].imshow(x_gen_store[i,(row*num_columns)+col,0,:,:21],vmin=(x_gen_store[i,:,0,:,:21]).min(), vmax=(x_gen_store[i,:,0,:,:21]).max()))
                        return plots
                    #print("x_gen shape:", x_gen_store.shape)
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])
                    img_path = save_dir + '/images/'
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    ani.save(img_path + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
                    plt.close('all')
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"/model_{ep}.pth")
            print('saved model at ' + save_dir + f"/model_{ep}.pth")


if __name__ == "__main__":
    train_cmapss()