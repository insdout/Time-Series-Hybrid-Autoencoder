import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import torch.nn.functional as F
from models import Encoder, Decoder, RVE
from collections import defaultdict
from torch import optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from IPython.display import clear_output
import random

class CMAPSS:
    def __init__(self,
                 dataset_name,
                 max_rul=150,
                 window_size=30,
                 sensors=['s_3', 's_4', 's_7', 's_11', 's_12'],
                 train_size=0.8,
                 alpha=0.1,
                 dir_path='./CMAPSSData/',
                 train_batch_size=100,
                 test_batch_size=64,
                 val_batch_size=64,
                 train_shuffle=True,
                 test_shuffle=False,
                 val_shuffle=False,
                 num_workers=2
                 ):

        self.dataset_name = dataset_name
        self.max_rul = max_rul
        self.window_size = window_size
        self.sensors = sensors
        self.train_size = train_size
        self.dir_path = dir_path
        self.alpha = alpha
        self.scaler = {}

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size
        self.train_shuffle = train_shuffle
        self.test_shuffle = test_shuffle
        self.val_shuffle = val_shuffle
        self.num_workers = num_workers

    def get_rul(self, df, final_rul):
        # Get the total number of cycles for each unit
        grouped_by_unit = df.groupby(by="unit_nr")
        max_cycle = grouped_by_unit["time_cycles"].max() + final_rul

        # Merge the max cycle back into the original frame
        result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

        # Calculate remaining useful life for each row
        remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
        result_frame["RUL"] = remaining_useful_life
        result_frame["RUL"] = result_frame["RUL"].astype(int)

        # drop max_cycle as it's no longer needed
        result_frame = result_frame.drop("max_cycle", axis=1)
        return result_frame

    def exponential_smoothing(self, df, sensors, n_samples, alpha=0.4):
        df = df.copy()
        # first, take the exponential weighted mean
        df[sensors] = df.groupby('unit_nr', group_keys=True)[sensors].apply(
            lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)

        # second, drop first n_samples of each unit_nr to reduce filter delay
        def create_mask(data, samples):
            result = np.ones_like(data)
            result[0:samples] = 0
            return result

        mask = df.groupby('unit_nr')['unit_nr'].transform(create_mask, samples=n_samples).astype(bool)
        df = df[mask]
        return df

    def add_operating_condition(self, df):
        df_op_cond = df.copy()

        df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
        df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))

        # converting settings to string and concatanating makes the operating condition into a categorical variable
        df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                                df_op_cond['setting_2'].astype(str) + '_' + \
                                df_op_cond['setting_3'].astype(str)
        return df_op_cond

    def condition_scaler(self, df):
        # apply operating condition specific scaling
        sensor_names = self.sensors

        def is_fit_called(obj):
            return hasattr(obj, "n_features_in_")

        for condition in df['op_cond'].unique():
            scaler = self.scaler.get(condition, StandardScaler())
            if not is_fit_called(scaler):
                #print(f"Fit scaler on: {condition}")
                scaler.fit(df.loc[df['op_cond'] == condition, sensor_names])
                self.scaler[condition] = scaler
            df.loc[df['op_cond'] == condition, sensor_names] = scaler.transform(
                df.loc[df['op_cond'] == condition, sensor_names])
        return df

    def _load_data(self):
        dir_path = self.dir_path
        dataset_name = self.dataset_name
        max_rul = self.max_rul
        # columns
        index_names = ['unit_nr', 'time_cycles']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ['s_{}'.format(i + 1) for i in range(0, 21)]
        col_names = index_names + setting_names + sensor_names
        # remove unused sensors
        drop_sensors = [element for element in sensor_names if element not in self.sensors]

        train_file = 'train_' + dataset_name + '.txt'

        # data readout
        train = pd.read_csv((dir_path + train_file), sep=r'\s+', header=None,
                            names=col_names)

        # create RUL values according to the piece-wise target function
        train_final_rul = np.zeros(train['unit_nr'].nunique())
        train = self.get_rul(train, train_final_rul)
        train['RUL'].clip(upper=max_rul, inplace=True)
        train = train.drop(drop_sensors, axis=1)

        test_file = 'test_' + dataset_name + '.txt'
        # data readout
        test = pd.read_csv((dir_path + test_file), sep=r'\s+', header=None,
                           names=col_names)
        y_test = pd.read_csv((dir_path + 'RUL_' + dataset_name + '.txt'), sep=r'\s+', header=None,
                             names=['RemainingUsefulLife'])
        test_final_rul = y_test.values.squeeze()
        # create RUL values according to the piece-wise target function
        test = self.get_rul(test, test_final_rul)
        test['RUL'].clip(upper=max_rul, inplace=True)
        test = test.drop(drop_sensors, axis=1)

        train = self.exponential_smoothing(train, self.sensors, 0, self.alpha)
        test = self.exponential_smoothing(test, self.sensors, 0, self.alpha)

        train = self.add_operating_condition(train)
        test = self.add_operating_condition(test)

        train = self.condition_scaler(train)
        x_test = self.condition_scaler(test)

        #Val split:
        gss = GroupShuffleSplit(n_splits=1, train_size=self.train_size, random_state=42)
        train_unit, val_unit = next(gss.split(train['unit_nr'].unique(), groups=train['unit_nr'].unique()))
        x_train = train[train['unit_nr'].isin(train_unit)]
        x_val = train[train['unit_nr'].isin(val_unit)]

        return x_train, x_test, x_val

    def get_datasets(self):
        dataset_kwargs = {"max_rul": self.max_rul, "window_size": self.window_size, "sensors": self.sensors}
        train_df, test_df, val_df = self._load_data()
        train_dataset = RVEDataset(dataset=train_df, mode='train', **dataset_kwargs)
        test_dataset = RVEDataset(dataset=test_df, mode='test', **dataset_kwargs)
        val_dataset = RVEDataset(dataset=val_df, mode='train', **dataset_kwargs)
        return train_dataset, test_dataset, val_dataset

    def get_dataloaders(self):
        train_kwargs = {
            "batch_size": self.train_batch_size,
            "shuffle": self.train_shuffle,
            "num_workers": self.num_workers
        }
        test_kwargs = {
            "batch_size": self.test_batch_size,
            "shuffle": self.test_shuffle,
            "num_workers": self.num_workers
        }
        val_kwargs = {
            "batch_size": self.val_batch_size,
            "shuffle": self.val_shuffle,
            "num_workers": self.num_workers
        }
        train_dataset, test_dataset, val_dataset = self.get_datasets()
        train_loader = DataLoader(dataset=train_dataset, **train_kwargs)
        test_loader = DataLoader(dataset=test_dataset, **test_kwargs)
        val_loader = DataLoader(dataset=val_dataset, **val_kwargs)
        return train_loader, test_loader, val_loader


class RVEDataset(Dataset):
    def __init__(self,
                 dataset,
                 mode='train',
                 max_rul=150,
                 window_size=30,
                 sensors=['s_3', 's_4', 's_7', 's_11', 's_12'],
                 ):
        """
        Dateset class, performs data preprocessing: normalization, sliding window slicing,
        handcrafted feature extraction, redundant features dropping
        :param data_path: string, path to data file
        :param mode: string, "train" or "test"
        :param rul_path: string, path to the RUL data file
        :param max_rul: int, maximal RUL, value at which RUL becomes constant
        :param window_size: int, width of sliding window
        :param standardize: bool, if set data will be standardized
        :param handcrafted: bool, if set handcrafted features will be extracted
        :param drop_features: list, list of feature indices to be dropped
        """
        self.final_rul = None
        self.run_id = None
        self.sequences = None
        self.targets = None
        self.ids = None
        self.mode = mode
        self.max_rul = max_rul
        self.window_size = window_size
        self.sensors = sensors
        self.df = dataset

        self.get_sequences(self.df)

    def get_sequences(self, df):
        window_size = self.window_size
        sensors = self.sensors
        columns_to_pick = ["unit_nr"] + sensors + ["RUL"]
        units = df['unit_nr'].unique()
        self.ids = units
        temp_sequences = []
        for unit in units:
            unit_df = df[df['unit_nr'] == unit].sort_values(by='time_cycles', ascending=True)
            unit_slice = unit_df[columns_to_pick].values
            slice_len = unit_slice.shape[0]
            if slice_len >= window_size:
                if self.mode == "train":
                    for i in range(0, slice_len - window_size + 1):
                        temp_sequences.append(unit_slice[i:i + window_size])
                else:
                    temp_sequences.append(unit_slice[slice_len - window_size:])
            else:
                #if self.mode == "train":
           	# row number < sequence length, only one sequence
            	# pad width first time-cycle value
            	temp_sequences.append(np.pad(unit_slice, ((window_size - slice_len, 0), (0, 0)), 'edge'))
            data = np.stack(temp_sequences)

        self.sequences = data[:, :, 1:-1]
        self.targets = data[:, -1, -1]
        self.run_id = data[:, 0, 0]

    def __getitem__(self, item):
        return torch.FloatTensor(self.sequences[item]), torch.FloatTensor([self.targets[item]])

    def __len__(self):
        return len(self.sequences)

    def get_run(self, run_id):
        mask = self.run_id == run_id
        return torch.FloatTensor(self.sequences[mask]), torch.FloatTensor(self.targets[mask])


class Trainer:

    @staticmethod
    def score(y, y_hat):
        score = 0
        y = y.cpu()
        y_hat = y_hat.cpu()
        for i in range(len(y_hat)):
            if y[i] <= y_hat[i]:
                score += np.exp(-(y[i] - y_hat[i]) / 10.0) - 1
            else:
                score += np.exp((y[i] - y_hat[i]) / 13.0) - 1
        return score

    @staticmethod
    def kl_loss(mean, log_var):
        loss = (-0.5 * (1 + log_var - mean ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        return loss

    @staticmethod
    def reg_loss(y, y_hat):
        # loss = F.mse_loss(y, y_hat, reduction='none')
        # loss = torch.mean(loss)
        return nn.MSELoss()(y, y_hat)

    @staticmethod
    def reconstruction_loss(x, x_hat):
        batch_size = x.shape[0]
        loss = F.mse_loss(x, x_hat, reduction='none')
        loss = loss.view(batch_size, -1).sum(axis=1)
        loss = loss.mean()
        return loss

    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, verbose=False):
        self.train_loader = train_loader
        self.test_loader = test_loader
        if val_loader is not None:
            self.val_loader = val_loader
            self.validate = True
        else:
            self.validate = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.history = defaultdict(list)
        self.epochs = 0
        self.verbose = verbose
        self.best_score = float('inf')
        self.best_rmse = float('inf')

    def train(self, n_epoch):
        for i in range(n_epoch):
            self.train_epoch()
            self.epochs += 1
            if self.verbose:
                plot_learning_curves(self.history, reconstruction=True)
                print(f"Epoch:{self.epochs} ")
                print(f"Train loss: {round(self.history['train_loss'][-1], 2)} "
                      f"kl loss: {round(self.history['train_kl_loss'][-1], 2)} "
                      f"reg loss: {round(self.history['train_reg_loss'][-1], 2)} "
                      f"recon loss: {round(self.history['train_recon_loss'][-1], 2)}")
                print(f"Valid loss: {round(self.history['valid_loss'][-1], 2)} "
                      f"kl loss: {round(self.history['valid_kl_loss'][-1], 2)} "
                      f"reg loss: {round(self.history['valid_reg_loss'][-1], 2)} "
                      f"recon loss: {round(self.history['valid_recon_loss'][-1], 2)}")
                print("Test:")
                print(f"     RMSE: {round(self.history['test_rmse'][-1], 2)} "
                      f"     Score: {round(self.history['test_score'][-1], 2)}"
                      )
                print(f"Best RMSE: {round(self.best_rmse, 2)} Best score: {round(self.best_score, 2)}")

    def train_epoch(self):
        epoch_loss = 0
        kl_loss_ep = 0
        reg_loss_ep = 0
        recon_loss_ep = 0
        recon_loss = 0
        self.model.train()
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            if self.model.decode_mode:
                y_hat, z, mean, log_var, x_hat = self.model(x)
                kl_loss = Trainer.kl_loss(mean, log_var)
                reg_loss = Trainer.reg_loss(y, y_hat)
                recon_loss = Trainer.reconstruction_loss(x, x_hat)
                loss = kl_loss + reg_loss + recon_loss

            else:
                y_hat, z, mean, log_var = self.model(x)
                kl_loss = Trainer.kl_loss(mean, log_var)
                reg_loss = Trainer.reg_loss(y, y_hat)
                loss = kl_loss + reg_loss

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(y)
            kl_loss_ep += kl_loss.item() * len(y)
            reg_loss_ep += reg_loss.item() * len(y)
            if self.model.decode_mode:
                recon_loss_ep += recon_loss.item() * len(y)

        self.history['train_loss'].append(epoch_loss / len(self.train_loader.dataset))
        self.history['train_kl_loss'].append(kl_loss_ep / len(self.train_loader.dataset))
        self.history['train_reg_loss'].append(reg_loss_ep / len(self.train_loader.dataset))
        self.history['train_recon_loss'].append(recon_loss_ep / len(self.train_loader.dataset))
        if self.validate:
            self.valid_epoch()
            self.test_epoch()

    def valid_epoch(self):
        epoch_loss = 0
        epoch_score = 0
        kl_loss_ep = 0
        reg_loss_ep = 0
        recon_loss_ep = 0
        recon_loss = 0
        self.model.eval()
        for batch_idx, data in enumerate(self.val_loader):
            with torch.no_grad():
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                if self.model.decode_mode:
                    y_hat, z, mean, log_var, x_hat = self.model(x)
                    kl_loss = Trainer.kl_loss(mean, log_var)
                    reg_loss = Trainer.reg_loss(y, y_hat)
                    recon_loss = Trainer.reconstruction_loss(x, x_hat)
                    loss = kl_loss + reg_loss + recon_loss
                else:
                    y_hat, z, mean, log_var = self.model(x)
                    kl_loss = Trainer.kl_loss(mean, log_var)
                    reg_loss = Trainer.reg_loss(y, y_hat)
                    loss = kl_loss + reg_loss

                epoch_loss += loss.item() * len(y)
                kl_loss_ep += kl_loss.item() * len(y)
                reg_loss_ep += reg_loss.item() * len(y)
                if self.model.decode_mode:
                    recon_loss_ep += recon_loss.item()

        epoch_loss = epoch_loss
        self.history['valid_loss'].append(epoch_loss / len(self.val_loader.dataset))
        self.history['valid_kl_loss'].append(kl_loss_ep / len(self.val_loader.dataset))
        self.history['valid_reg_loss'].append(reg_loss_ep / len(self.val_loader.dataset))
        self.history['valid_recon_loss'].append(recon_loss_ep / len(self.val_loader.dataset))

    def test_epoch(self):
        epoch_rmse = 0
        epoch_score = 0
        self.model.eval()
        for batch_idx, data in enumerate(self.test_loader):
            with torch.no_grad():
                x, y = data
                x, y = x.to(self.device), y.to(self.device)

                y_hat, *_ = self.model(x)
                loss = F.mse_loss(y_hat, y)

                epoch_rmse += loss.item() * len(y)
                epoch_score += Trainer.score(y, y_hat).item()

        epoch_rmse = (epoch_rmse / len(self.test_loader.dataset)) ** 0.5
        self.best_score = min(self.best_score, epoch_score)
        self.best_rmse = min(self.best_rmse, epoch_rmse)
        self.history['test_rmse'].append(epoch_rmse)
        self.history['test_score'].append(epoch_score)


def viz_latent_space(model, data, targets=[], title='Final', save=False, show=True):
    data = torch.tensor(data).float()
    model.to('cpu')
    with torch.no_grad():
        z, _, _  = model.encoder(data)
        z = z.numpy()
    plt.figure(figsize=(8, 4))
    if len(targets)>0:
        plt.scatter(z[:, 0], z[:, 1], c=targets, s=1.5)
    else:
        plt.scatter(z[:, 0], z[:, 1])
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    plt.colorbar()
    plt.title(title)
    if show:
        plt.tight_layout()
    if save:
        plt.savefig('./images/latent_space_epoch'+str(title)+'.png')


def get_trainer(dataset_name, sensors, max_rul=125, alpha=0.1, hidden_size=200, latent_dim=2, num_layers=1,
                batch_size=128, lr=0.0005, window_size=30, reconstruct=False):
    input_size = len(sensors)

    train_loader, test_loader, val_loader = CMAPSS(
        dataset_name=dataset_name,
        max_rul=max_rul,
        train_batch_size=batch_size,
        window_size=window_size,
        sensors=sensors,
        alpha=alpha
    ).get_dataloaders()

    x_train = train_loader.dataset.sequences
    y_train = train_loader.dataset.targets

    x_val = val_loader.dataset.sequences
    y_val = val_loader.dataset.targets

    x_test = test_loader.dataset.sequences
    y_test = test_loader.dataset.targets

    model = get_RVE_model(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        window_size=window_size,
        num_layers=num_layers,
        reconstruct=reconstruct
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer, True)

    return trainer, x_train, y_train, x_val, y_val, x_test, y_test, train_loader, test_loader, val_loader


def plot_learning_curves(history, reconstruction=False):
    clear_output(True)
    if reconstruction:
        fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(16, 6))

        ax[0][0].plot(history['train_loss'], label='train loss')
        ax[0][0].plot(history['valid_loss'], label='val loss')

        ax[0][1].plot(history['train_kl_loss'], label='train kl_loss loss')
        ax[0][1].plot(history['valid_kl_loss'], label='val kl_loss loss')

        ax[1][0].plot(history['train_reg_loss'], label='train reg loss')
        ax[1][0].plot(history['valid_reg_loss'], label='val reg loss')

        ax[1][1].plot(history['train_recon_loss'], label='train reconstruction loss')
        ax[1][1].plot(history['valid_recon_loss'], label='val reconstruction loss')

        ax[2][0].plot(history['test_rmse'], label='test rmse')
        ax[2][1].plot(history['test_score'], label='test score')

        ax[0][0].legend(loc='upper right')
        ax[0][1].legend(loc='upper right')
        ax[1][0].legend(loc='upper right')
        ax[1][1].legend(loc='upper right')
        ax[2][0].legend(loc='upper right')
        ax[2][1].legend(loc='upper right')
        ax[0][0].grid(True)
        ax[0][1].grid(True)
        ax[1][0].grid(True)
        ax[1][1].grid(True)
        ax[2][0].grid(True)
        ax[2][1].grid(True)
        ax[0][0].set_yscale('log')
        ax[0][1].set_yscale('log')
        ax[1][0].set_yscale('log')
        ax[1][1].set_yscale('log')
        ax[2][0].set_yscale('log')
        ax[2][1].set_yscale('log')
        plt.show()


def get_RVE_model(
        input_size,
        hidden_size,
        latent_dim,
        window_size,
        num_layers=1,
        reconstruct=False):
    enc_block = Encoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        num_layers=num_layers
    )

    dec_block = Decoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        window_size=window_size,
        num_layers=num_layers
    )

    model = RVE(enc_block, dec_block, reconstruct=reconstruct)
    return model


def get_engine_runs(dataloader, model):
    engine_ids = dataloader.dataset.ids
    history = defaultdict(dict)
    model.eval().to('cpu')

    for engine_id in engine_ids:
        with torch.no_grad():
            x, y = dataloader.dataset.get_run(engine_id)
            y_hat, z, *_ = model(x)
            history[engine_id]['rul'] = y.numpy()
            history[engine_id]['rul_hat'] = y_hat.numpy()
            history[engine_id]['z'] = z.numpy()

    return history


def plot_engine_run(history, engine_id=None):
    engine_ids = history.keys()

    if engine_id is None:
        engine_id = random.choice(list(engine_ids))

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(12, 6))
    real_rul = history[engine_id]['rul']
    rul_hat = history[engine_id]['rul_hat']
    ax[0].plot(real_rul)
    ax[0].plot(rul_hat)
    ax[0].set_title(f"Engine Unit #{engine_id}")
    ax[0].set_xlabel("Time(Cycle)")
    ax[0].set_ylabel("RUL")
    for run in engine_ids:
        z = history[run]['z']
        targets = history[run]['rul']
        pa = ax[1].scatter(z[:, 0], z[:, 1], c=targets, s=1.5)
    cba = plt.colorbar(pa, shrink=1.0)
    cba.set_label("RUL")

    z = history[engine_id]['z']
    targets = history[engine_id]['rul']
    pb = ax[1].scatter(z[:, 0], z[:, 1], c=targets, s=15, cmap=plt.cm.gist_heat_r)
    cbb = plt.colorbar(pb, shrink=1.0)
    cbb.set_label(f"Engine #{engine_id} RUL")
    ax[1].set_xlabel("z - dim 1")
    ax[1].set_ylabel("z - dim 2")
    plt.show()


if __name__ == "__main__":
    ds = CMAPSS(dataset_name="FD001")
    x_train, x_test, x_val = ds.get_datasets()
    print(len(x_train), len(x_val), len(x_test))
    x_train, x_test, x_val = ds.get_dataloaders()
    print(len(x_train.dataset), len(x_val.dataset), len(x_test.dataset))
