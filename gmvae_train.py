import torch
import torch.optim as optim
import numpy as np
import json
import logging
import os
from utils.gmvae_loss import TotalLoss
from utils.gmvae_utils import get_model, NumpyEncoder
from utils.metric_dataloader import MetricDataPreprocessor
import hydra
from hydra.utils import instantiate
from collections import defaultdict


log = logging.getLogger(__name__)


class Trainer:
    @staticmethod
    def score(y, y_hat):
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

    def __init__(self, model, optimizer, criterion, train_loader,
                 test_loader, val_loader, device, path, track_ids=False, tracked_ids={},
                 n=1):
        """
        Trainer class for training and evaluating a PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to train and _evaluate.
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            criterion (callable): The loss function to compute the training loss.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            device (str): Device to run the computations on (e.g., "cuda" or "cpu").
            track_ids (bool): Flag indicating whether to track specific sample IDs during training (default: True).
            tracked_ids (set): Set of sample IDs to track during training (default: empty set).
            n (int): Number of sample IDs to track during training (default: 2).
            transform_fn (callable): Optional function to transform the input data (default: flatten_mnist).
        """
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.history = defaultdict(list)
        self.track_ids = track_ids
        self.tracked_ids = tracked_ids
        self.n = n
        self.ids_history = defaultdict(dict)

        self.current_epoch = 0
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")
        self.model = model.to(self.device)

        self.path = path

    def train(self, epochs):
        """
        Train the model for the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train the model.
        """
        log.info(f"Training on {self.device}")
        if self.track_ids:
            if len(self.tracked_ids) == 0:
                self.tracked_ids = self._get_n_ids_per_class(self.n)
            self._get_tracked_x_true()

        for epoch in range(epochs):
            self._train_epoch()
            self._evaluate()

            # Track history for ids over epochs:
            if self.track_ids:
                self._infer_tracked_ids()

            train_loss = self.history["train_loss"][-1]
            train_score = self.history["train_score"][-1]
            train_rmse = self.history["train_rmse"][-1]
            test_loss = self.history["test_loss"][-1]
            test_score = self.history["test_score"][-1]
            test_rmse = self.history["test_rmse"][-1]

            log.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                     f"Test Loss: {test_loss:.4f} Train score: {train_score:.4f} Test score: {test_score:.4f}, "
                     f"Train RMSE {train_rmse:.4f} Test RMSE: {test_rmse:.4f}")
        # After trainig actions:
        os.makedirs(self.path, exist_ok=True)
        history_path = os.path.join(self.path, "history.json")
        self.dump_to_json(self.history, history_path, indent=4)
        ids_history_path = os.path.join(self.path, "ids_history.json")
        self.dump_to_json(self.ids_history, ids_history_path)

    def _train_epoch(self):
        """
        Training loop
        """
        model = self.model.train()
        optimizer = self.optimizer
        criterion = self.criterion
        dataloader = self.train_loader
        device = self.device

        running_loss = 0.0
        running_entropy = 0.0
        pred_labels = []
        true_labels = []

        for batch_idx, data in enumerate(dataloader):
            batch_len = len(dataloader.dataset)
            pairs_mode = dataloader.dataset.return_pairs

            x, true_rul = data
            x, true_rul = x.to(self.device), true_rul.to(self.device)

            optimizer.zero_grad()

            out_dict = model(x)
            # Computing the total loss:
            loss = criterion(x, true_rul, out_dict)

            loss['total_loss'].backward()
            
            # Clip gradients by their norm
            max_norm = 1.0  # Set the maximum gradient norm threshold
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            running_loss += loss['total_loss'].item()
            running_entropy += loss["cond_entropy"].item()

            pred_labels.extend(out_dict["rul_hat"].detach().cpu().numpy())
            true_labels.extend(true_rul.detach().cpu().numpy())

        pred_labels = torch.tensor(np.array(pred_labels))
        true_labels = torch.tensor(np.array(true_labels))
        rmse = torch.nn.functional.mse_loss(pred_labels, true_labels).item()
        score = self.score(true_labels, pred_labels).item()
        train_loss = running_loss / len(dataloader.dataset)
        cond_entropy = running_entropy / len(dataloader.dataset)
        self.history["train_loss"].append(train_loss)
        self.history["train_cond_entropy"].append(-cond_entropy)
        self.history["train_rmse"].append(rmse)
        self.history["train_score"].append(score)
        self.current_epoch += 1

    def _evaluate(self):
        """
        Training loop
        """
        model = self.model.eval()
        criterion = self.criterion
        dataloader = self.test_loader
        device = self.device

        running_loss = 0.0
        running_entropy = 0.0
        pred_labels = []
        true_labels = []

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                batch_len = len(dataloader.dataset)
                pairs_mode = dataloader.dataset.return_pairs

                x, true_rul = data
                x, true_rul = x.to(self.device), true_rul.to(self.device)

                out_dict = model(x)
                # Computing the total loss:
                loss = criterion(x, true_rul, out_dict)

                running_loss += loss['total_loss'].item()
                running_entropy += loss["cond_entropy"].item()

                pred_labels.extend(out_dict["rul_hat"].detach().cpu().numpy())
                true_labels.extend(true_rul.detach().cpu().numpy())
        self.history["rul_hat"].append(pred_labels)
        self.history["true_rul"].append(true_labels)
        pred_labels = torch.tensor(np.array(pred_labels))
        true_labels = torch.tensor(np.array(true_labels))
        rmse = torch.nn.functional.mse_loss(pred_labels, true_labels).item()
        score = self.score(true_labels, pred_labels).item()

        test_loss = running_loss / len(dataloader.dataset)
        cond_entropy = running_entropy / len(dataloader.dataset)
        self.history["test_loss"].append(test_loss)
        self.history["test_cond_entropy"].append(-cond_entropy)
        self.history["test_rmse"].append(rmse)
        self.history["test_score"].append(score)

    def _get_n_ids_per_class(self, n):
        """_summary_

        Args:
            n (_type_): _description_

        Returns:
            _type_: _description_
        """
        targets = self.test_loader.dataset.targets
        data_len = len(targets)

        # random_indices = torch.tensor(random_indices)
        random_indices = np.random.choice(list(range(data_len)), 10)
        return random_indices

    def _get_tracked_x_true(self):
        """
        """
        for true_id in self.tracked_ids:
            true_id = int(true_id)
            self.ids_history[true_id]["x_true"] = self.test_loader.dataset.sequences[true_id]

    def _infer_tracked_ids(self):
        """_summary_
        """
        model = self.model.eval()
        ids = self.tracked_ids
        device = self.device

        for batch_idx, data in enumerate(self.train_loader):

            x, true_rul = data
            x, true_rul = x.to(self.device), true_rul.to(self.device)

            out_dict = model.infer(x)

            for rel_id, true_id in enumerate(ids):
                true_id = int(true_id)
                for key in out_dict.keys():
                    temp_array = out_dict[key][rel_id].detach().cpu().numpy()
                    self.ids_history[true_id].setdefault(key, []).append(temp_array)

    def dump_to_json(self, data, file_path, indent=None):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, cls=NumpyEncoder)
        log.info(f"JSON data saved to: {file_path}")


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    # current hydra output folder
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    # Preparing dataloaders:
    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    # Instantiating the model:
    model = get_model(**config.model)

    # Instantiating the optimizer:
    optimizer = instantiate(config.optimizer, params=model.parameters())

    criterion = TotalLoss(k=config.model.k)

    device = "cuda"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        device=device,
        path=output_dir
        )
    trainer.train(100)


if __name__ == "__main__":
    main()
