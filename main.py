from metric_dataloader import MetricDataPreprocessor
from train import Trainer
from test import Tester
from loss import TotalLoss
import torch


import hydra 
from hydra.utils import instantiate

import os.path




@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):

    #Train
    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    encoder = instantiate(config.model.encoder)
    decoder = instantiate(config.model.decoder)
    model_rve = instantiate(config.model.rve, encoder=encoder, decoder=decoder)

    optimizer = instantiate(config.optimizer,  params=model_rve.parameters())

    total_loss = TotalLoss(config)

    trainer = Trainer(**config.trainer,
        model=model_rve, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        total_loss=total_loss, 
        )
    trainer.train()

    # Test
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    path = hydra_cfg['runtime']['output_dir']
    model = torch.load(path+"/rve_model.pt")

    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    tester = Tester(path, model, val_loader, test_loader)
    z, t, y = tester.get_z()
    print(z.shape, y.shape, y.shape, len(val_loader.dataset))
    print(tester.get_test_score())
    tester.test()

if __name__ == "__main__":
    main()
