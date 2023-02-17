import torch
from torch.utils.data import Dataset
from scipy import interpolate
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class RVEDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 mode='train',
                 max_rul=150,
                 window_size=30,
                 standardize=False,
                 sensors=['s_3', 's_4', 's_7', 's_11', 's_12'],
                 alpha=0.1,
                 dir_path='./CMAPSSData/',
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
        self.dataset_name = dataset_name
        self.max_rul = max_rul
        self.window_size = window_size
        self.sensors = sensors
        self.dir_path = dir_path
        self.alpha = alpha
        self.scaler = {}
        self.df = self.get_dataset()

    def get_rul(self, df):
        final_rul = self.final_rul
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

    def get_dataset(self):
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
        if self.mode == 'train':
            train_file = 'train_' + dataset_name + '.txt'

            # data readout
            train = pd.read_csv((dir_path + train_file), sep=r'\s+', header=None,
                                names=col_names)

            # create RUL values according to the piece-wise target function
            self.final_rul = np.zeros(train['unit_nr'].nunique())
            train = self.get_rul(train)
            train['RUL'].clip(upper=max_rul, inplace=True)
            data = train.drop(drop_sensors, axis=1)
        else:
            test_file = 'test_' + dataset_name + '.txt'
            # data readout
            test = pd.read_csv((dir_path + test_file), sep=r'\s+', header=None,
                               names=col_names)
            y_test = pd.read_csv((dir_path + 'RUL_' + dataset_name + '.txt'), sep=r'\s+', header=None,
                                 names=['RemainingUsefulLife'])
            self.final_rul = y_test.values.squeeze()
            # create RUL values according to the piece-wise target function
            test = self.get_rul(test)
            test['RUL'].clip(upper=max_rul, inplace=True)
            data = test.drop(drop_sensors, axis=1)
        data = self.exponential_smoothing(data, self.sensors, 0, self.alpha)
        data = self.add_operating_condition(data)
        data = self.condition_scaler(data, self.sensors)
        self.get_sequences(data)
        return data

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

    def condition_scaler(self, df, sensors):
        # apply operating condition specific scaling
        sensor_names = self.sensors

        def is_fit_called(obj):
            return hasattr(obj, "n_features_in_")

        for condition in df['op_cond'].unique():
            scaler = self.scaler.get(condition, StandardScaler())
            if not is_fit_called(scaler):
                scaler.fit(df.loc[df['op_cond'] == condition, sensor_names])
            df.loc[df['op_cond'] == condition, sensor_names] = scaler.transform(
                df.loc[df['op_cond'] == condition, sensor_names])

        return df

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


if __name__ == "__main__":
    dataset = RVEDataset("FD001")
    print(dataset.df)
    print()
    print(dataset.sequences.shape)
    print(dataset.targets.shape)
    print(dataset.run_id.shape)
    print()
    print(*map(lambda x: x.shape, dataset.get_run(1)))
    print(len(dataset))
    print(*map(lambda x: x.shape, next(iter(dataset))))

    dataset = RVEDataset("FD001", mode="test")
    print(dataset.df)
    print()
    print(dataset.sequences.shape)
    print(dataset.targets.shape)
    print(dataset.run_id.shape)
    print()
    print(*map(lambda x: x.shape, dataset.get_run(1)))
    print()
    """"
        unit_nr  time_cycles  setting_1  ...      s_12  RUL        op_cond
0            1            1        0.0  ...  0.334262  150  0.0_0.0_100.0
1            1            2        0.0  ...  1.174899  150  0.0_0.0_100.0
2            1            3        0.0  ...  1.364721  150  0.0_0.0_100.0
3            1            4        0.0  ...  1.961302  150  0.0_0.0_100.0
4            1            5        0.0  ...  1.052871  150  0.0_0.0_100.0

       unit_nr  time_cycles  setting_1  ...      s_12  RUL        op_cond
0            1            1        0.0  ...  0.271690  150  0.0_0.0_100.0
1            1            2        0.0  ...  0.805001  150  0.0_0.0_100.0
2            1            3        0.0  ...  1.066547  150  0.0_0.0_100.0
3            1            4        0.0  ...  1.405702  150  0.0_0.0_100.0
4            1            5        0.0  ...  1.340304  150  0.0_0.0_100.0
    """
