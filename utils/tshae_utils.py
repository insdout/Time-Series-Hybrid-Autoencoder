import torch
import os
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate


def load_tshae_model(model_path):
    
    dir_path = os.path.dirname(model_path)
    
    tshae_config_path = os.path.join(dir_path, ".hydra/config.yaml")
    tshae_config = OmegaConf.load(tshae_config_path)

    # Instantiating the model:
    encoder = instantiate(tshae_config.model.encoder)
    decoder = instantiate(tshae_config.model.decoder)

    model_rve = instantiate(tshae_config.model.tshae, encoder=encoder, decoder=decoder)

    model_rve.load_state_dict(torch.load(model_path))
    model_rve.eval()
    return model_rve
    

@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):

    path = config.diffusion.checkpoint_tshae.path
    model = load_tshae_model(path)
    print(model)
    
if __name__ == "__main__":
    main()
    