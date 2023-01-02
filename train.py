import torch
import numpy as np


class TrainUtil:
    def __init__(self, model, train_loader, test_loader, optimizer, verbosity=1, max_rul=150, handcrafted=False):
        """
        Trainer utility class, which performs model training and validation.
        :param model: object, model object
        :param train_loader: object, train_loader object
        :param test_loader: object, test_loader object
        :param optimizer: object, optimizer object
        :param verbosity: int, controls the verbosity of the console output
        :param max_rul: int, maximal RUL, after which RUL becomes constant
        :param handcrafted: bool, flag if set to True, handcrafted features will be extracted and passed to input
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.verbosity = verbosity
        self.history = {}
        self.criterion = torch.nn.MSELoss()
        self.max_rul = max_rul
        self.handcrafted = handcrafted
        if self.verbosity:
            print(f"Device: {self.device}")

    def train_one_epoch(self, train_loader=None):
        """
        Performs training and validation for one epoch
        :param train_loader: object, train_loader object (optional)
        :return: float, average loss on epoch
        """
        if not train_loader:
            train_loader = self.train_loader

        self.model.train()
        loss_acc = 0
        factor = self.max_rul
        for batch_index, data in enumerate(train_loader):
            self.optimizer.zero_grad()
            if self.handcrafted:
                x, hc, labels = data
                x, hc, labels = x.to(self.device), hc.to(self.device), labels.to(self.device)
                predictions = self.model(x, hc)
            else:
                x, labels = data
                x, labels = x.to(self.device), labels.to(self.device)
                predictions = self.model(x)

            loss = self.criterion(factor * predictions, factor * labels)
            loss_acc += loss.item() * len(labels)
            loss.backward()
            self.optimizer.step()

        train_loss = (loss_acc / len(train_loader.dataset)) ** 0.5
        if self.verbosity and self.verbosity > 2:
            print(f"Train average loss: {train_loss}")
        return train_loss

    def validate(self, test_loader=None):
        """
        Performs validation on test dateset
        :param test_loader: object, test_loader object (optional)
        :return: tuple of float, average validation loss and score
        """
        if not test_loader:
            test_loader = self.test_loader
        self.model.eval()
        loss_acc = 0
        score_acc = 0
        factor = self.max_rul
        for batch_index, data in enumerate(test_loader):
            with torch.no_grad():
                if self.handcrafted:
                    x, hc, labels = data
                    x, hc, labels = x.to(self.device), hc.to(self.device), labels.to(self.device)
                    predictions = self.model(x, hc)
                else:
                    x, labels = data
                    x, labels = x.to(self.device), labels.to(self.device)
                    predictions = self.model(x)

                loss = self.criterion(factor * predictions, factor * labels)
                score = self.score(factor * predictions, factor * labels)
                loss_acc += loss.item() * len(labels)
                score_acc += score.item()

        val_loss = (loss_acc / len(test_loader.dataset)) ** 0.5
        if self.verbosity and self.verbosity > 2:
            print(f"Validation average loss: {val_loss} average score: {score_acc}")
        return val_loss, score_acc

    def train(self, epoch, train_loader=None, test_loader=None):
        """
        Trains the model on given number of epochs
        :param epoch: int, number of epochs
        :param train_loader: object, train_loader object (optional)
        :param test_loader: object, test_loader object (optional)
        :return: dict, history of train, validation loss and scores
        """
        if not train_loader:
            train_loader = self.train_loader
        if not test_loader:
            test_loader = self.test_loader
        history = {"train_loss": [], "val_loss": [], "val_score": []}
        for epoch_num in range(epoch):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_score = self.validate(test_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_score"].append(val_score)
            if self.verbosity:
                print(f"Epoch: {epoch_num + 1} train loss: {train_loss :.3f} "
                      f"val loss: {val_loss :.3f} score: {val_score :.3f}")
        return history

    @staticmethod
    def score(y_true, y_hat):
        """
        Calculates the score given true and predicted y
        :param y_true: tensor, true y
        :param y_hat: tensor, predicted y
        :return: float, average score
        """
        score = 0
        y_true = y_true.cpu()
        y_hat = y_hat.cpu()
        for i in range(len(y_hat)):
            if y_true[i] <= y_hat[i]:
                score += np.exp(-(y_true[i] - y_hat[i]) / 10.0) - 1
            else:
                score += np.exp((y_true[i] - y_hat[i]) / 13.0) - 1
        return score
