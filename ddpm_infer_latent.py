from utils.ddpm_utils import load_latent_trajectory
from utils.ddpm_utils import plot_results_reconstruction
import torch
from models.ddpm_models import ContextUnet, DDPM
from omegaconf import OmegaConf
from utils.metric_dataloader import MetricDataPreprocessor
from utils.ddpm_utils import get_diffusion_outputs_from_z
from utils.plot_utils import plot_engine_run_diff
from utils.plot_utils import plot_engine_run_diff_decision_boundary
from utils.plot_utils import reconstruct_and_plot
from utils.tshae_utils import load_tshae_model
import hydra
import pickle
import os


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    
    extrapolated_z_path = config.diffusion.extrapolated_latent.path
    extrapolated_z = load_latent_trajectory(extrapolated_z_path)
    print(extrapolated_z.keys())
    
    tshae_checkpoint_path = config.diffusion.checkpoint_tshae.path
    print(tshae_checkpoint_path)
    tshae_config_path = os.path.dirname(tshae_checkpoint_path) + "/.hydra/config.yaml"
    tshae_config = OmegaConf.load(tshae_config_path)
    
    preproc = MetricDataPreprocessor(**tshae_config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()
    print(f"train set: {len(train_loader.dataset)} val set: {len(val_loader.dataset)}")
    model_tshae = load_tshae_model(tshae_checkpoint_path)
    #print(model_tshae)
    
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
    model_tshae.eval().to(device)

    engine_runs = get_diffusion_outputs_from_z(
        z_space_dict=extrapolated_z,
        tshae_model=model_tshae,
        diffusion_model=ddpm,
        dataloader=val_loader,
        num_samples=config.diffusion.diffusion_tester.num_samples,
        w=config.diffusion.diffusion_tester.w,
        quantile=config.diffusion.diffusion_tester.quantile,
        mode=config.diffusion.diffusion_tester.mode
        )

    pickle_path = os.path.join(output_dir, "engine_runs_diff.pickle")
    with open(pickle_path, 'wb') as handle:
        pickle.dump(engine_runs, handle)
    
    plot_results_reconstruction(engine_runs, output_dir)

if __name__ == "__main__":
    main()