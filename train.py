import torch
import numpy as np


class TrainUtil:
    def __init__(self, model, train_loader, test_loader, optimizer, verbosity=1, max_rul=150):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.verbosity = verbosity
        self.history = {}
        self.criterion = torch.nn.MSELoss()
        self.max_rul = max_rul
        print(f"Device: {self.device}")

    def train_one_epoch(self, train_loader=None):
        if not train_loader:
            train_loader = self.train_loader

        self.model.train()
        loss_acc = 0
        factor = self.max_rul
        for batch_index, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss = self.criterion(factor*predictions, factor*labels)
            loss_acc += loss.item()*len(labels)
            loss.backward()
            self.optimizer.step()

        train_loss = (loss_acc/len(train_loader.dataset))**0.5
        if self.verbosity:
            print(f"Train average loss: {train_loss}")
        return train_loss

    def validate(self, test_loader=None):
        if not test_loader:
            test_loader = self.test_loader
        self.model.eval()
        loss_acc = 0
        score_acc = 0
        factor = self.max_rul
        for batch_index, data in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predictions = self.model(inputs)
                loss = self.criterion(factor*predictions, factor*labels)
                score = self.score(factor*predictions, factor*labels)
                loss_acc += loss*len(labels)
                score_acc += score.item()

        val_loss = (loss_acc/len(test_loader.dataset))**0.5
        if self.verbosity:
            print(f"Validation average loss: {val_loss} average score: {score_acc}")
        return val_loss, score_acc

    def train(self, epoch, train_loader=None, test_loader=None):
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
            print(f"Epoch: {epoch_num+1} train loss: {train_loss :.3f} "
                  f"val loss: {val_loss :.3f} score: {val_score :.3f}")
        return history


    @staticmethod
    def score(y_true, y_hat):
        score = 0
        y_true = y_true.cpu()
        y_hat = y_hat.cpu()
        for i in range(len(y_hat)):
            if y_true[i] <= y_hat[i]:
                score += np.exp(-(y_true[i] - y_hat[i]) / 10.0) - 1
            else:
                score += np.exp((y_true[i] - y_hat[i]) / 13.0) - 1
        return score


