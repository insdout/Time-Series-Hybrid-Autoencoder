import torch
import numpy  as np
from utils.plot_utils import plot_engine_run_diff, plot_engine_run_diff_decision_boundary, reconstruct_and_plot
from models.ddpm_models import ContextUnet, DDPM
from omegaconf import OmegaConf
from utils.metric_dataloader import MetricDataPreprocessor
import hydra
import pickle
import os


def get_engine_runs_diffusion(
        dataloader, 
        rve_model, 
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
    rve_model.eval().to(device)
    diffusion_model.eval().to(device)

    for engine_id in engine_ids:
        engine_id =int(engine_id)
        with torch.no_grad():
            x, y = dataloader.dataset.get_run(engine_id)
            x = x.to(device)
            y = y.to(device)

            #print(f" xshape: {x.shape} y shape: {y.shape}")
            y_hat, z, *_ = rve_model(x)

            num_z = z.shape[0]
            x_hat, _ = diffusion_model.sample_cmapss(n_sample=num_samples, size=(1,32,32), device=x.device, z_space_contexts=z, guide_w = w)
            
            #TODO: get several samples and pick the best one
            #================================================
            z = z.unsqueeze(1)

            x_hat = x_hat.squeeze(1)[:,:,:21]
            with torch.no_grad(): 
                rul_hat_diff, z_diff, *_  = rve_model(x_hat)
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
            
            x_diff_samples = x_hat.reshape(num_samples, num_z, 32, 21).permute(1,0,2,3)

            x_diff_samples = x_diff_samples[range(x_diff_samples.shape[0]), best_samples]
            z_diff = z_diff[range(z_diff.shape[0]), best_samples]
            rul_hat_diff = rul_hat_diff[range(rul_hat_diff.shape[0]), best_samples]
            '''
            print(f"final x_hat shape: {x_diff_samples.shape}")
            print("final z", z_diff.shape)
            print("final rul", rul_hat_diff.shape)
            '''

            history[engine_id]['rul'] = y.detach().cpu().numpy()
            history[engine_id]['rul_hat'] = y_hat.detach().cpu().numpy()
            history[engine_id]['z'] = z.squeeze().detach().cpu().numpy()
            history[engine_id]["x"] = x.detach().cpu().numpy()
            history[engine_id]["x_hat"] = x_diff_samples.detach().cpu().numpy()
            history[engine_id]["rul_hat_diff"] = rul_hat_diff.detach().cpu().numpy()
            history[engine_id]["z_diff"] = z_diff.detach().cpu().numpy()

    return history



@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def test(config):

    tshae_checkpoint_path = config.diffusion.checkpoint_tshae.path
    print(tshae_checkpoint_path)
    tshae_config_path = os.path.dirname(tshae_checkpoint_path) + "/.hydra/config.yaml"
    tshae_config = OmegaConf.load(tshae_config_path)
    
    preproc = MetricDataPreprocessor(**tshae_config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()
    print(f"train set: {len(train_loader.dataset)} val set: {len(val_loader.dataset)}")
    model_rve = torch.load(tshae_checkpoint_path)
    #print(model_rve)
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    print(f"output dir: {output_dir}")

    ddpm_checkpoint_path = config.diffusion.checkpoint_ddpm.path
    ddpm_config_path = os.path.dirname(ddpm_checkpoint_path) + "/.hydra/config.yaml"
    ddpm_checkpoint_config = OmegaConf.load(ddpm_config_path)
 
    n_T = ddpm_checkpoint_config.diffusion.ddpm_train.n_T # 500
    device = ddpm_checkpoint_config.diffusion.ddpm_train.device #"cuda:0" or "cpu"#
    z_dim   = ddpm_checkpoint_config.diffusion.ddpm_train.z_dim
    n_feat = ddpm_checkpoint_config.diffusion.ddpm_train.n_feat # 128 ok, 256 better (but slower)
    drop_prob = ddpm_checkpoint_config.diffusion.ddpm_model.drop_prob
    
    ddpm = DDPM(
        nn_model=ContextUnet(
        in_channels=1, 
        n_feat=n_feat, 
        z_dim=z_dim), 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=drop_prob)
    ddpm.load_state_dict(torch.load(config.diffusion.checkpoint_ddpm.path))
    ddpm.eval().to(device)
    model_rve.eval().to(device)

    val_ids = val_loader.dataset.ids
    print(val_loader.dataset.ids)
    engine_runs = get_engine_runs_diffusion(
        rve_model=model_rve, 
        diffusion_model=ddpm, 
        dataloader=val_loader,
        num_samples=config.diffusion.diffusion_tester.num_samples,
        w=config.diffusion.diffusion_tester.w,
        quantile=config.diffusion.diffusion_tester.quantile,
        mode=config.diffusion.diffusion_tester.mode
        )
    
    with open(output_dir + "engine_runs_diff.pickle", 'wb') as handle:
        pickle.dump(engine_runs, handle)
    for engine in engine_runs.keys():
        plot_engine_run_diff(engine_runs,engine_id=engine, img_path=output_dir, save=True)

if __name__ == "__main__":
    test()