import torch
import numpy  as np
from utils.plot_utils import plot_engine_run_diff, plot_engine_run_diff_decision_boundary, reconstruct_and_plot
from models.ddpm_models import ContextUnet, DDPM
from utils.metric_dataloader import MetricDataPreprocessor
from utils.tshae_utils import load_tshae_model
import hydra
import pickle


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def plot(config):

    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()
    print(f"train set: {len(train_loader.dataset)} val set: {len(val_loader.dataset)}")
    model_tshae = load_tshae_model(config.diffusion.checkpoint.path)
    #print(model_tshae)
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    print(f"output dir: {output_dir}")
    
    with open("engine_runs_diff.pickle", 'rb') as handle:
        engine_runs = pickle.load(handle)
    for engine in engine_runs.keys():
        plot_engine_run_diff(
            engine_runs, 
            engine_id=engine, 
            img_path=output_dir, 
            save=True, 
            show=False
            )
        plot_engine_run_diff_decision_boundary(
            model_tshae, 
            engine_runs, 
            img_path=output_dir, 
            engine_id=engine, 
            title="engine_run_boundary", 
            save=True, 
            show=False)

if __name__ == "__main__":
    plot()