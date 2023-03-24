import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import random

import hydra 
from omegaconf import DictConfig, OmegaConf


class MetricDataPreprocessor:
    """TODO: docstring"""
    def __init__(self,
                 dataset_name,
                 max_rul,
                 window_size,
                 sensors,
                 train_size,
                 alpha,
                 dir_path
                 ):

        self.dataset_name = dataset_name
        self.max_rul = max_rul
        self.window_size = window_size
        if type(sensors) != list:
            self.sensors = list(sensors)
        else:
            self.sensors = sensors
        self.train_size = train_size
        self.dir_path = dir_path
        self.alpha = alpha
        self.scaler = {}


    def _get_rul(self, df, final_rul):
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

    def _exponential_smoothing(self, df, sensors, n_samples, alpha=0.4):
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

    def _add_operating_condition(self, df):
        df_op_cond = df.copy()

        df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
        df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))

        # converting settings to string and concatanating makes the operating condition into a categorical variable
        df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                                df_op_cond['setting_2'].astype(str) + '_' + \
                                df_op_cond['setting_3'].astype(str)
        return df_op_cond

    def _condition_scaler(self, df):
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
        train = self._get_rul(train, train_final_rul)
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
        test = self._get_rul(test, test_final_rul)
        test['RUL'].clip(upper=max_rul, inplace=True)
        test = test.drop(drop_sensors, axis=1)

        train = self._exponential_smoothing(train, self.sensors, 0, self.alpha)
        test = self._exponential_smoothing(test, self.sensors, 0, self.alpha)

        train = self._add_operating_condition(train)
        test = self._add_operating_condition(test)

        train = self._condition_scaler(train)
        x_test = self._condition_scaler(test)

        #Val split:
        gss = GroupShuffleSplit(n_splits=1, train_size=self.train_size, random_state=42)
        train_unit, val_unit = next(gss.split(train['unit_nr'].unique(), groups=train['unit_nr'].unique()))
        x_train = train[train['unit_nr'].isin(train_unit)]
        x_val = train[train['unit_nr'].isin(val_unit)]

        return x_train, x_test, x_val

    def get_datasets(self, train_dataset_kwargs, test_dataset_kwargs, val_dataset_kwargs):
        dataset_kwargs = {"max_rul": self.max_rul, "window_size": self.window_size, "sensors": self.sensors}
        train_df, test_df, val_df = self._load_data()
        train_dataset = MetricDataset(dataset=train_df, mode='train', **dataset_kwargs, **train_dataset_kwargs)
        test_dataset = MetricDataset(dataset=test_df, mode='test', **dataset_kwargs, **test_dataset_kwargs)
        val_dataset = MetricDataset(dataset=val_df, mode='train', **dataset_kwargs, **val_dataset_kwargs)
        return train_dataset, test_dataset, val_dataset
    
    def get_dataloaders(self, 
                        train_dataset_kwargs, 
                        test_dataset_kwargs, 
                        val_dataset_kwargs, 
                        train_dataloader_kwargs, 
                        test_dataloader_kwargs, 
                        val_dataloader_kwargs
                        ):

        train_dataset, test_dataset, val_dataset = self.get_datasets(train_dataset_kwargs, test_dataset_kwargs, val_dataset_kwargs)
        train_loader = DataLoader(dataset=train_dataset, **train_dataloader_kwargs)
        test_loader = DataLoader(dataset=test_dataset, **test_dataloader_kwargs)
        val_loader = DataLoader(dataset=val_dataset, **val_dataloader_kwargs)
        return train_loader, test_loader, val_loader


class MetricDataset(Dataset):
    def __init__(self,
                 dataset,
                 mode, 
                 max_rul,
                 window_size,
                 sensors,
                 return_pairs,
                 range_divisors
                 ):
        """
        TODO: change docstring
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
        self.return_pairs = return_pairs
        if self.return_pairs:
            self.range_low = (0, range_divisors[0])
            self.rande_middle = (range_divisors[0]+1, range_divisors[1]-1)
            self.range_high = (range_divisors[1], max_rul)
        self.final_rul = None
        self.run_id = None
        self.sequences = None
        self.targets = None
        self.ids = None
        self.mode = mode
        self.max_rul = max_rul
        self.window_size = window_size
        if type(sensors) != list:
            self.sensors = list(sensors)
        else:
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
    
    def __getitem__(self, index):
        if self.return_pairs:
            return self.get_pairs(index)
        return self.sequences[index], self.targets[index]

    def get_pairs(self, index):
        run_id = self.run_id[index]
        rul = self.targets[index]
        pairs_x, pairs_y = self.get_three_pairs(run_id)

        if rul > 120:
            pairs_x[0, 0] = self.sequences[index]
            pairs_y[0, 0] = self.targets[index]
        elif rul < 5:
            pairs_x[1, 0] = self.sequences[index]
            pairs_y[1, 0] = self.targets[index]
        else:
            pairs_x[2, 0] = self.sequences[index]
            pairs_y[2, 0] = self.targets[index]
        return pairs_x, pairs_y

    def __len__(self):
        return len(self.sequences)
    
    def get_sample(self, run_id, rul_range):
        mask = (self.targets >= rul_range[0]) & (self.targets <= rul_range[0]) & (self.run_id == run_id)
        indexes = mask.nonzero()
        idx = random.choice(indexes)
        return self.sequences[idx], self.targets[idx]
    
    def get_three_pairs(self, run_id):
        x = []
        y =[]
        for range in [self.range_high, self.rande_middle, self.range_low]:
            x1, y1 = self.get_sample(run_id, range)
            x2, y2 = self.get_sample(run_id, range)
            x.append([x1.squeeze(), x2.squeeze()])
            y. append([y1.squeeze(), y2.squeeze()])
        return np.array(x), np.array(y)

    def get_run(self, run_id):
        mask = self.run_id == run_id
        return torch.FloatTensor(self.sequences[mask]), torch.FloatTensor(self.targets[mask])


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    print()
    print("==========================")
    print(config.keys())
    print(config.data_preprocessor)
    print(type(config.data_preprocessor.sensors) != list)
    print(type([]) == list)
    print(config.train_dataset)
    print(config.test_dataset)
    print(config.val_dataset)
    print("==========================")

    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders(
        config.train_dataset, 
        config.test_dataset, 
        config.val_dataset, 
        config.train_dataloader, 
        config.test_dataloader, 
        config.val_dataloader
        )
    x, y = train_loader.dataset[0]
    print(f"x shape: {x.shape} y shape: {y.shape}")


if __name__ == "__main__":
    main()
