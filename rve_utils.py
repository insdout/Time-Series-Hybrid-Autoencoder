import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy import interpolate
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit


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
                print(f"Fit scaler on: {condition}")
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
                if self.mode == "train":
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


class ModelTrainer:
    @staticmethod
    def score(y_hat, y):
        if torch.is_tensor(y_hat):
            y_hat = y_hat.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()

        error = y_hat - y
        pos_e = np.exp(-error[error < 0] / 13) - 1
        neg_e = np.exp(error[error >= 0] / 10) - 1
        return sum(pos_e) + sum(neg_e)


    def __init__(self, model, optimizer, criterion, train_loader, test_loader, val_loader=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        if val_loader:
            self.val_loader = val_loader
            self.validate = True
        else:
            self.validate = False
        self.criterion = criterion
        self.history = defaultdict(list)
        print("dataset len", len(self.train_loader.dataset), len(self.test_loader.dataset))
        print("Training on:", self.device)

    def train(self, n_epoch):
        for i in range(n_epoch):
            self.train_epoch()
            print(f"Epoch:{i} "
                  f"Train Loss: {round(self.history['train_loss'][-1], 2)} "
                  f"Test Loss: {round(self.history['test_loss'][-1], 2)} "
                  f"Test Score: {round(self.history['test_score'][-1], 2)}"
                  )

    def train_epoch(self):
        epoch_loss = 0
        self.model.train()
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            x, y, _ = data
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()*len(y)

        epoch_loss = (epoch_loss/len(self.train_loader.dataset))**0.5
        self.history['train_loss'].append(epoch_loss)
        self.test_valid_epoch(mode='test')
        if self.validate:
            self.test_valid_epoch(mode='validation')

    def test_valid_epoch(self, mode):
        assert mode in ['test', 'validation'], 'wrong mode'
        if mode == 'test':
            data_loader = self.test_loader
        else:
            data_loader = self.val_loader
        epoch_loss = 0
        epoch_score = 0
        self.model.eval()
        for batch_idx, data in enumerate(data_loader):
            with torch.no_grad():
                x, y, _ = data
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

                epoch_loss += loss.item() * len(y)
                epoch_score += ModelTrainer.score(y_hat, y)

        epoch_loss = (epoch_loss / len(data_loader.dataset)) ** 0.5
        self.history[f'{mode}_loss'].append(epoch_loss)
        self.history[f'{mode}_score'].append(epoch_score)


    def save_history(self):
        pass


if __name__ == "__main__":
    ds = CMAPSS(dataset_name="FD001")
    x_train, x_test, x_val = ds.get_datasets()
    print(len(x_train), len(x_val), len(x_test))
    x_train, x_test, x_val = ds.get_dataloaders()
    print(len(x_train.dataset), len(x_val.dataset), len(x_test.dataset))
