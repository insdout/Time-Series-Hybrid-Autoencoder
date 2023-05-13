import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import random

import hydra
from omegaconf import  OmegaConf


class MetricDataPreprocessor:
    """
    CMAPSS dataset class, handles data loading, preprocessing and sliding window slicing.
    Splits preprocessed data into train and validation datasets.
    Generates Train, Test, Validation datasets and dataloaders.
    """

    def __init__(self,
                dataset_name,
                max_rul,
                window_size,
                sensors,
                train_size,
                alpha,
                dir_path,
                fix_seed=True,
                downsample_healthy_train=False,
                downsample_healthy_validation=False,
                downsample_healthy_test=False,
                downsample_healthy_train_p=0,
                downsample_healthy_validation_p=0,
                downsample_healthy_test_p=0,
                downsample_rul_threshold_train=None,
                downsample_rul_threshold_validation=None,
                downsample_rul_threshold_test=None,
                train_ds_mode="train",
                train_ds_return_pairs=True,
                train_ds_eps=3,
                train_ds_max_eps=6,
                train_ds_triplet_healthy_rul=120,
                test_ds_mode="test",
                test_ds_return_pairs=False,
                test_ds_eps=3,
                test_ds_max_eps=6,
                test_ds_triplet_healthy_rul=120,
                val_ds_mode="train",
                val_ds_return_pairs=True,
                val_ds_eps=3,
                val_ds_max_eps=6,
                val_ds_triplet_healthy_rul=120,
                train_dl_batch_size=100,
                train_dl_shuffle=True,
                train_dl_num_workers=2,
                test_dl_batch_size=100,
                test_dl_shuffle=False,
                test_dl_num_workers=2,
                val_dl_batch_size=100,
                val_dl_shuffle=True,
                val_dl_num_workers=2
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
        self.fix_seed = fix_seed
        self.downsample_healthy_train = downsample_healthy_train
        self.downsample_healthy_validation = downsample_healthy_validation
        self.downsample_healthy_test = downsample_healthy_test
        self.downsample_healthy_train_p = downsample_healthy_train_p
        self.downsample_healthy_validation_p = downsample_healthy_validation_p
        self.downsample_healthy_test_p = downsample_healthy_test_p
        self.downsample_rul_threshold_train=downsample_rul_threshold_train,
        self.downsample_rul_threshold_validation=downsample_rul_threshold_validation,
        self.downsample_rul_threshold_test=downsample_rul_threshold_test,
        self.train_ds_kwargs = {
            "mode": train_ds_mode,
            "return_pairs": train_ds_return_pairs,
            "triplet_eps": train_ds_eps,
            "triplet_max_eps": train_ds_max_eps,
            "triplet_healthy_rul": train_ds_triplet_healthy_rul,
            "downsample_healthy": downsample_healthy_train,
            "downsample_healthy_p": downsample_healthy_train_p,
            "downsample_rul_threshold": downsample_rul_threshold_train
        }
        self.test_ds_kwargs = {
            "mode": test_ds_mode,
            "return_pairs": test_ds_return_pairs,
            "triplet_eps": test_ds_eps,
            "triplet_max_eps": test_ds_max_eps,
            "triplet_healthy_rul": test_ds_triplet_healthy_rul,
            "downsample_healthy": downsample_healthy_test,
            "downsample_healthy_p": downsample_healthy_test_p,
            "downsample_rul_threshold": downsample_rul_threshold_test
        }
        self.val_ds_kwargs = {
            "mode": val_ds_mode,
            "return_pairs": val_ds_return_pairs,
            "triplet_eps": val_ds_eps,
            "triplet_max_eps": val_ds_max_eps,
            "triplet_healthy_rul": val_ds_triplet_healthy_rul,
            "downsample_healthy": downsample_healthy_validation,
            "downsample_healthy_p": downsample_healthy_validation_p,
            "downsample_rul_threshold": downsample_rul_threshold_validation
        }
        self.train_dl_kwargs = {
            "batch_size": train_dl_batch_size,
            "shuffle": train_dl_shuffle,
            "num_workers": train_dl_num_workers
        }
        self.test_dl_kwargs = {
            "batch_size": test_dl_batch_size,
            "shuffle": test_dl_shuffle,
            "num_workers": test_dl_num_workers
        }
        self.val_dl_kwargs = {
            "batch_size": val_dl_batch_size,
            "shuffle": val_dl_shuffle,
            "num_workers": val_dl_num_workers
        }

    def _get_rul(self, df, final_rul):
        """
        Groups dataset by unit_nr (unit number) and calculates RUL within each group.
        In case of test dataset, where run stops not at RUL=0, final_rul is the RUL value of the last datapoint in group
        :param df: DataFrame
        :param final_rul: int
        :return: DataFrame
        """
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
        """
        Performs exponential smoothing of the input data
        via https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
        If alpha = 1, no smoothing will be applied.
        :param df: DataFrame
        :param sensors: number of sensor features, int
        :param n_samples:
        :param alpha: smoothing factor [0, 1]
        :return: DataFrame
        """
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
        """
        Joins 2 setting columns in one string which uniquely identifies the operational settings
        :param df: DataFrame
        :return: DataFrame
        """
        df_op_cond = df.copy()

        df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
        df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))

        # converting settings to string and concatenating makes the operating condition into a categorical variable
        df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' \
                                + df_op_cond['setting_2'].astype(str) + '_' \
                                + df_op_cond['setting_3'].astype(str)
        return df_op_cond

    def _condition_scaler(self, df):
        """
        Applies Scaler for each operational condition separately.
        :param df: DataFrame
        :return: DataFrame
        """
        # apply operating condition specific scaling
        sensor_names = self.sensors

        def is_fit_called(obj):
            return hasattr(obj, "n_features_in_")

        for condition in df['op_cond'].unique():
            scaler = self.scaler.get(condition, StandardScaler())
            # Check if the Scaler for this condition is fitted and fit if it is not:
            if not is_fit_called(scaler):
                scaler.fit(df.loc[df['op_cond'] == condition, sensor_names])
                self.scaler[condition] = scaler
            df.loc[df['op_cond'] == condition, sensor_names] = scaler.transform(
                df.loc[df['op_cond'] == condition, sensor_names])
        return df

    def _load_data(self):
        """
        Loading data from csv files.
        :return: Train, Test, Validation DataFrames
        """
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

        # Validation split:
        group_split = GroupShuffleSplit(n_splits=1, train_size=self.train_size, random_state=42)
        train_unit, val_unit = next(group_split.split(train['unit_nr'].unique(), groups=train['unit_nr'].unique()))
        x_train = train[train['unit_nr'].isin(train_unit)]
        x_val = train[train['unit_nr'].isin(val_unit)]

        return x_train, x_test, x_val

    def get_datasets(self):
        """
        Calls data loading method and return 3 DataSets.
        :return: Train, Test, Validation Datasets
        """
        dataset_kwargs = {"max_rul": self.max_rul, "window_size": self.window_size, "sensors": self.sensors}
        train_df, test_df, val_df = self._load_data()
        train_dataset = MetricDataset(dataset=train_df, **dataset_kwargs, **self.train_ds_kwargs)
        test_dataset = MetricDataset(dataset=test_df, **dataset_kwargs, **self.test_ds_kwargs)
        val_dataset = MetricDataset(dataset=val_df, **dataset_kwargs, **self.val_ds_kwargs)
        return train_dataset, test_dataset, val_dataset

    def get_dataloaders(self):
        """
        Calls dataset generation method and return 3 DataSets.
        :return: Train, Test, Validation DataLoaders
        """

        # see https://pytorch.org/docs/stable/notes/randomness.html
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        if self.fix_seed:
            g = torch.Generator()
            g.manual_seed(0)
        else:
            seed_worker = None
            g = None
        
        print(f"fix dataloader: {self.fix_seed} seed_worker: {seed_worker} g: {g}")

        train_dataset, test_dataset, val_dataset = self.get_datasets()
        train_loader = DataLoader(dataset=train_dataset, worker_init_fn=seed_worker, generator=g, **self.train_dl_kwargs)
        test_loader = DataLoader(dataset=test_dataset, worker_init_fn=seed_worker, generator=g, **self.test_dl_kwargs)
        val_loader = DataLoader(dataset=val_dataset, worker_init_fn=seed_worker, generator=g, **self.val_dl_kwargs)
        return train_loader, test_loader, val_loader


class MetricDataset(Dataset):
    def __init__(self,
                 dataset,
                 mode,
                 max_rul,
                 window_size,
                 sensors,
                 return_pairs,
                 triplet_eps,
                 triplet_max_eps,
                 triplet_healthy_rul,
                 downsample_healthy,
                 downsample_healthy_p,
                 downsample_rul_threshold=None
                 ):

        self.return_pairs = return_pairs
        self.eps = triplet_eps
        self.max_eps = triplet_max_eps
        self.healthy_rul = triplet_healthy_rul
        self.final_rul = None
        self.run_id = None
        self.sequences = None
        self.targets = None
        self.ids = None
        self.mode = mode
        self.max_rul = max_rul
        self.window_size = window_size
        self.downsample_healthy = downsample_healthy
        self.downsample_healthy_p = downsample_healthy_p
        self.downsample_rul_threshold = downsample_rul_threshold
        if type(sensors) != list:
            self.sensors = list(sensors)
        else:
            self.sensors = sensors
        self.df = dataset

        self.get_sequences(self.df)

    def get_sequences(self, df):
        """
        Splits each engine run into moving window slices
        If the length of engine run is less than window size, the data will be padded
        :param df: DataFrame
        """
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
                # if self.mode == "train":
                # row number < sequence length, only one sequence
                # pad width first time-cycle value
                temp_sequences.append(np.pad(unit_slice, ((window_size - slice_len, 0), (0, 0)), 'edge'))
            data = np.stack(temp_sequences)

        self.sequences = data[:, :, 1:-1]
        self.targets = data[:, -1, -1]
        self.run_id = data[:, 0, 0]

    def __getitem__(self, index):
        """
        Returns sliding window datapoint and target at given index. If option return_pairs is True,
        3 datapoints will be returned for Triplet loss (datapoint at given index: anchor, positive and negative
        datapoint.
        :param index: index of datapoint to be returned, int
        :return: 2 or 6 torch.FloatTensors
        """
        if self.downsample_healthy:
            run_id = self.run_id[index]
            rul = self.targets[index]
            downsampe_rul_threshold = self.max_rul
            if self.downsample_rul_threshold:
                downsampe_rul_threshold = self.downsample_rul_threshold
            if (rul >= self.max_rul) & (np.random.rand(1) < self.downsample_healthy_p):
                candidate_point_mask = (self.targets < downsampe_rul_threshold) & (self.run_id == run_id)
                candidate_point_mask_indexes = np.flatnonzero(candidate_point_mask)
                idx = random.choice(candidate_point_mask_indexes)
                index = idx
            else:
                index = index
        if self.return_pairs:
            return self.get_triplet(index)
        return torch.FloatTensor(self.sequences[index]), torch.FloatTensor([self.targets[index]])

    def get_triplet(self, index):
        """
        Returns datapoint at given index (Anchor), positive and negative datapoints
        :param index: index of datapoint to be returned, int
        :return: 6 torch.FloatTensors
        """
        run_id = self.run_id[index]
        rul = self.targets[index]
        x, y = torch.FloatTensor(self.sequences[index]), torch.FloatTensor([self.targets[index]])
        pos_x, pos_y = self.get_positive_sample(run_id, rul)
        neg_x, neg_y = self.get_negative_sample(run_id, rul)

        return x, pos_x, neg_x, y, pos_y, neg_y

    def __len__(self):
        """
        Returns length of the dataset.
        :return: int
        """
        return len(self.sequences)

    def get_positive_sample(self, run_id, rul):
        """
        Selects positive datapoint within datapoints of the current unit number, which satisfy condition:
        RUL of datapoint is within +/- eps from RUL of the RUL of the given datapoint
        :param run_id: unit number of provided datapoint
        :param rul: RUL of provided datapoint
        :return: 2 torch.FloatTensors
        """
        if rul >= self.healthy_rul:
            candidate_point_mask = (self.targets >= self.healthy_rul) & (self.run_id == run_id)
        else:
            candidate_point_mask = (abs(self.targets - rul) <= self.eps) & (self.run_id == run_id)
        candidate_point_mask_indexes = np.flatnonzero(candidate_point_mask)
        idx = random.choice(candidate_point_mask_indexes)
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])

    def get_negative_sample(self, run_id, rul):
        """
        Selects negative datapoint within datapoints of the current unit number, which satisfy condition:
            RUL of datapoint is further than +/- eps
            AND
            RUL of datapoint is within +/- max_eps
        from RUL of the given datapoint
        :param run_id: unit number of provided datapoint
        :param rul: RUL of provided datapoint
        :return: 2 torch.FloatTensors
        """
        if rul >= self.healthy_rul:
            candidate_point_mask = (self.targets >= self.healthy_rul - self.eps) \
                                   & (self.targets < self.healthy_rul) & \
                                   (self.run_id == run_id)
        else:
            candidate_point_mask = (abs(self.targets - rul) <= self.max_eps) \
                                   & (abs(self.targets - rul) >= self.eps) & \
                                   (self.run_id == run_id)
        candidate_point_mask_indexes = np.flatnonzero(candidate_point_mask)
        idx = random.choice(candidate_point_mask_indexes)
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])

    def get_run(self, run_id):
        """
        Returns all datapoints within given unit number (run_id)
        :param run_id: unit number
        :return:  2 torch.FloatTensors
        """
        mask = self.run_id == run_id
        return torch.FloatTensor(self.sequences[mask]), torch.FloatTensor(self.targets[mask])


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    print()
    print("==========================")
    print(OmegaConf.to_yaml(config))
    print()
    print(config)
    print(config.model.input_size)
    print("keys:", config.keys())
    print()
    print("conf:", config.data_preprocessor)
    print()
    print("keys.preprocessor", config.data_preprocessor.sensors, type(config.data_preprocessor.sensors),
          config.data_preprocessor.sensors[0])
    print()
    print(type(config.data_preprocessor.sensors) != list)
    print()
    print(type([]) == list)
    print("==========================")

    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()
    x, pos_x, neg_x, y, pos_y, neg_y = next(iter(train_loader))
    print(f"x shape: {x.shape} y shape: {y.shape} {pos_x.shape}{pos_y.shape}{neg_y.shape}{neg_y.shape}")


if __name__ == "__main__":
    main()
