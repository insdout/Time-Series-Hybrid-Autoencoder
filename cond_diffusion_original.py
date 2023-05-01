''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import random
import hydra
from metric_dataloader import MetricDataPreprocessor

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        #print("     UP shapes:", x.shape, skip.shape)
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, z_dim=2):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.z_dim  = z_dim 

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(z_dim , 2*n_feat)
        self.contextembed2 = EmbedFC(z_dim , 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        #print("===============================")
        #print("Forward | shape x:", x.shape)
        x = self.init_conv(x)
        #print("Forward | shape init conv:", x.shape)
        down1 = self.down1(x)
        #print("Forward | shape down1:", down1.shape)
        down2 = self.down2(down1)
        #print("Forward | shape down2:", down2.shape)
        hiddenvec = self.to_vec(down2)
        #print("Forward | shape hiddenvec:", hiddenvec.shape)
        #print("===============================")
        # convert context to one hot embedding
 
        
        # mask out context if context_mask == 1
        #print("===============================")
        #print("cm shape 1:", context_mask.shape)
        #print("cm shape 2:", context_mask.shape, "z_dim:", self.z_dim)
        context_mask = context_mask.repeat(1, self.z_dim)
        #print("cm shape 3:", context_mask.shape)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        #print("c: ", c)
        #print("context mask: ", context_mask)
        c = c * context_mask
        #print("c after: ", c)
        #print("===============================")
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)
        #print("===============================")
        #print("Forward UP| shape hiddenvec:", hiddenvec.shape)
        up1 = self.up0(hiddenvec)
        #print("Forward UP| shape up1:", up1.shape)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        #print("temb1", temb1.shape)
        #print("cemb1", cemb1.shape)
        #print("up1", up1.shape)
        #print(cemb1*up1+ temb1)
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        #print("Forward UP| shape up2:", up2.shape)
        up3 = self.up2(cemb2*up2+ temb2, down1)
        #print("Forward UP| shape up3:", up3.shape)
        out = self.out(torch.cat((up3, x), 1))
        #print("Forward UP| shape out:", out.shape)
        #print("===============================")
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        #print("_ts shape:", _ts.shape)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        #print("x_t shape:", x_t.shape)
        # dropout context with some probability
        batch_size, z_dim = c.shape
        context_mask = torch.bernoulli(torch.zeros((batch_size, 1))+self.drop_prob).to(self.device)
        #print("train context mask mean", context_mask.mean())
        #print("TRUE CONTEXT MASK:", context_mask)
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))
    
    def sample_cmapss(self, n_sample, size, device, z_space_contexts, guide_w = 0.0):
        #size = (1, 28, 28)
        # z_space_contexts = (N, z_dim) = (N, 2)
        num_z_contexts, z_dim = z_space_contexts.shape

        x_i = torch.randn(n_sample*num_z_contexts, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = z_space_contexts.to(device) # latent space vectors
        #print("x_i shape: ", x_i.shape)
        #print("c_i shape: ", c_i.shape)
        c_i = c_i.repeat(n_sample, 1)
        #print("c_i repeated shape: ", c_i.shape)

        num_inputs, z_dim = c_i.shape
        # don't drop context at test time
        context_mask = torch.zeros((num_inputs, 1)).to(device)
        #print("context_mask shape: ", context_mask.shape)
        
        # double the batch
        c_i = c_i.repeat(2, 1)
        #print("c_i doubled shape: ", c_i.shape)
        context_mask = context_mask.repeat(2, 1)
        #print("context_mask doubled shape: ", context_mask.shape)
        context_mask[num_inputs:] = 1. # makes second half of batch context free
        #print("context mask mean", context_mask.mean())

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            #print(f'sampling timestep {i}')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample*num_z_contexts,1,1,1)
           

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample*num_z_contexts, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            #import os, psutil; 
            #print(round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3,2), "GB")
            #print("devices:", x_i.device, c_i.device, t_is.device, context_mask.device)
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            #print("model done")
            #print("eps shape", eps.shape)
            eps1 = eps[:n_sample*num_z_contexts]
            eps2 = eps[n_sample*num_z_contexts:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample*num_z_contexts]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        #print("end sample cmapss")
        return x_i, x_i_store


    

@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def check(config):
    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    model_path = "./outputs/2023-04-22/15-07-36_windowsize_32/rve_model.pt"
    model_rve = torch.load(model_path).cpu()
    #print(model_rve)


    pairs_mode = train_loader.dataset.return_pairs
    if pairs_mode:
        x, pos_x, neg_x, true_rul, _, _ = next(iter(train_loader))
    
    input_rve = x[:2,:,:]
    predicted_rul, z, mean, log_var, x_hat = model_rve(input_rve)
    
    m = nn.ReplicationPad2d((0, 11, 0, 0))
    input_d = m(input_rve)

    print("INPUT SHAPES:", input_rve.shape, input_d.shape, z.shape)

  
    n_epoch = 1
    batch_size = 256
    n_T = 400 # 500
    device = "cpu"#"cuda:0"
    z_dim   = 2
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = './outputs/diffusion_outputs/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    
    ddpm = DDPM(
        nn_model=ContextUnet(
        in_channels=1, 
        n_feat=n_feat, 
        z_dim=z_dim), 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=0.1)
    
    loss = ddpm(input_d.unsqueeze(1), z)
    print(loss)
    print()
    print("===========================")
    print("SAMPLE:")
    print("===========================")
    z_samples = torch.rand((3, 2))
    ddpm.eval()
    with torch.no_grad():
        xi, xi_sore = ddpm.sample_cmapss(n_sample=2, size=(1,32,32), device="cpu", z_space_contexts=z_samples, guide_w = 0.0)
    print("done")
    print("generated shape: ", xi.shape)
    print("end")

@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def train_cmapss(config):

    preproc = MetricDataPreprocessor(**config.diffusion.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()
    print(f"train set: {len(train_loader.dataset)} val set: {len(val_loader.dataset)}")
    model_path = "./outputs/2023-04-22/15-07-36_windowsize_32/rve_model.pt"
    model_rve = torch.load(config.diffusion.checkpoint.path)
    #print(model_rve)

  
    n_epoch = 20
    batch_size = 256
    n_T = 500 # 500
    device = "cuda:0" #"cpu"#
    z_dim   = 2
    n_feat = 256 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    save_dir = './outputs/diffusion_outputs/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    
    ddpm = DDPM(
        nn_model=ContextUnet(
        in_channels=1, 
        n_feat=n_feat, 
        z_dim=z_dim), 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=0.1)

    ddpm.to(device)
    model_rve.eval().to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

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
                predicted_rul, z, mean, log_var, x_hat = model_rve(x)
                m = nn.ReplicationPad2d((0, 11, 0, 0))
                x_diffusion = m(x)

            optim.zero_grad()
            x_diffusion = x_diffusion.unsqueeze(1).to(device)
            context = z.to(device)
            #print(f"diffusion input shape:{x_diffusion.shape} context shape: {context.shape}")
            loss = ddpm(x_diffusion, context)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        rul_range = np.arange(0, 100, 10)
        run_ids = train_loader.dataset.ids
        idx = random.choice(run_ids)
        #print("run ids:", run_ids)
        run_x, run_rul = train_loader.dataset.get_run(idx)
        x_samples = run_x[torch.isin(run_rul, torch.Tensor(rul_range))]
        rul_seed = run_rul[torch.isin(run_rul, torch.Tensor(rul_range))]
        #print("run_rul", run_rul)
        #print("rul_range", torch.Tensor(rul_range))
        x_samples = x_samples.to(device)
        with torch.no_grad():
            predicted_rul, z_samples, mean, log_var, x_hat = model_rve(x_samples)


        #print("x_samples shape: ", x_samples.shape)
        #print("z_samples shape: ", z_samples.shape)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4 
            num_columns = z_samples.shape[0]
            num_rows = n_sample
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store =ddpm.sample_cmapss(n_sample=n_sample, size=(1,32,32), device=device, z_space_contexts=z_samples, guide_w = 0.0)

                # append some real images at bottom, order by class also
                x_real = m(x_samples).to(device)

                #print("x_all shape:",  x_real.unsqueeze(1).shape, "x_gen shape:", x_gen.shape)
                #print("x_gen_sore shape:", x_gen_store.shape)
                x_all = torch.cat([x_gen, x_real.unsqueeze(1)])
                #print("x_all shape:", x_all.shape)
                #grid = make_grid(x_all[:,:,:,:21], nrow=num_columns)
                #save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
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
                        # plots.append(axs[row, col].imshow(x_gen_store[i,(row*z_dim   )+col,0],cmap='gray'))
                        #print(i, row, col, row*num_columns+col)
                        #print(i, row, col, row*num_columns+col,"shape item:", x_gen_store[i,(row*num_columns)+col,0].shape)
                        axs[row, col].imshow(x_all[row*num_columns+col,:,:,:21].cpu().squeeze(),vmin=(x_all[:,:,:,:21].min()), vmax=(x_all[:,:,:,:21].max()))
                plt.savefig(save_dir + f"image_ep{ep}_w{w}.png", dpi=100)
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
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*z_dim   )+col,0],cmap='gray'))
                                #print(i, row, col, row*num_columns+col)
                                #print(i, row, col, row*num_columns+col,"shape item:", x_gen_store[i,(row*num_columns)+col,0].shape)
                                plots.append(axs[row, col].imshow(x_gen_store[i,(row*num_columns)+col,0,:,:21],vmin=(x_gen_store[i,:,0,:,:21]).min(), vmax=(x_gen_store[i,:,0,:,:21]).max()))
                        return plots
                    #print("x_gen shape:", x_gen_store.shape)
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
                    plt.close('all')
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")


if __name__ == "__main__":
    train_cmapss()
    #check()

