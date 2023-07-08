from utils.plot_utils import get_engine_runs
import json
import torch
import numpy as np
import hydra
from utils.metric_dataloader import MetricDataPreprocessor
from utils.tshae_utils import load_tshae_model
from omegaconf import OmegaConf
import random
import os
import random


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    """
    Main function for generating and saving engine run dictionaries based on the retrained TSHAE model.

    :param config: Configuration object created by Hydra.
    """
    
    tshae_checkpoint_path = config.diffusion.checkpoint_tshae.path
    print(tshae_checkpoint_path)
    tshae_config_path = os.path.dirname(tshae_checkpoint_path) + "/.hydra/config.yaml"
    tshae_config = OmegaConf.load(tshae_config_path)
    model_tshae = load_tshae_model(tshae_checkpoint_path)
    
    # fix random seeds:
    if config.random_seed.fix == True:

        torch.manual_seed(config.random_seed.seed)
        torch.cuda.manual_seed_all(config.random_seed.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True) 
        np.random.seed(config.random_seed.seed)
        random.seed(config.random_seed.seed)
        # see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()
    
    train_dict = get_engine_runs(train_loader, model_tshae)
    val_dict = get_engine_runs(val_loader, model_tshae)
    test_dict = get_engine_runs(test_loader, model_tshae)

    with open('./assets/train_dict_retrained.json', 'w') as fp:
        json.dump(train_dict, fp, cls=NumpyEncoder)

    with open('./assets/val_dict_retrained.json', 'w') as fp:
        json.dump(val_dict, fp, cls=NumpyEncoder)

    with open('./assets/test_dict_retrained.json', 'w') as fp:
        json.dump(test_dict, fp, cls=NumpyEncoder)
        
        
if __name__ == "__main__":
    main()
    