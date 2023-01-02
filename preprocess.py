import torch
from torch.utils.data import Dataset
from scipy import interpolate
import numpy as np


class CMAPSSSlidingWin(Dataset):
    def __init__(self,
                 data_path,
                 mode='train',
                 rul_path=None,
                 max_rul=150,
                 window_size=30,
                 standardize=False,
                 handcrafted=False,
                 drop_features=[5, 9, 10, 14, 20, 22, 23]
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
        self.data = np.loadtxt(fname=data_path, dtype=np.float32)
        if len(drop_features) > 0:
            self.data = np.delete(self.data, drop_features, axis=1)
        if rul_path:
            self.data_rul = np.loadtxt(fname=rul_path, dtype=np.float32)
        self.mode = mode
        self.max_rul = max_rul
        self.window_size = window_size
        self.num_runs = int(self.data[-1, 0])
        self.standardize = standardize
        self.handcrafted = handcrafted
        self.run_id = []

        if self.handcrafted:
            if mode == 'train':
                self.x, self.hc, self.y = self.prepare_train()
            elif mode == 'test':
                self.x, self.hc, self.y = self.prepare_test()
        else:
            if mode == 'train':
                self.x,  self.y = self.prepare_train()
            elif mode == 'test':
                self.x,  self.y = self.prepare_test()

    def prepare_train(self):
        """
        Normalizes data, applies sliding window slicing,
        extracts handcrafted features if self.handcrafted is True
        Handcrafted features are normalized.
        :return: tuple of numpy arrays x, hc, y
        """
        data = self.norm_data()
        x = []
        hc = []
        y = []
        for run in range(1, self.num_runs + 1):
            mask = np.asarray(data[:, 0] == run).nonzero()[0]
            temp_data = data[mask, 2:]

            for i in range(len(temp_data) - self.window_size + 1):
                frame = temp_data[i:i + self.window_size, :]
                rul = len(temp_data) - i - self.window_size
                x.append(frame)
                y.append(min(rul, self.max_rul))
                self.run_id.append(run)
        x = np.array(x)
        y = np.array(y) / self.max_rul
        self.run_id = np.array(self.run_id)

        if self.handcrafted:
            for i in range(len(x)):
                one_sample = x[i]
                hc.append(self.fea_extract(one_sample))

            mu = np.mean(hc, axis=0)
            sigma = np.std(hc, axis=0)
            eps = 1e-10
            hc = (hc - mu) / (sigma + eps)
            hc = np.array(hc)
            return x, hc, y
        else:
            return x, y

    def prepare_test(self):
        """
        Normalizes data, applies sliding window slicing,
        extracts handcrafted features if self.handcrafted is True
        Handcrafted features are normalized.
        :return: tuple of arrays x, hc, y
        :return: tuple of numpy arrays
        """
        data = self.norm_data()
        x = []
        hc = []
        y = []
        for run in range(1, self.num_runs + 1):
            mask = np.asarray(data[:, 0] == run).nonzero()[0]
            temp_data = data[mask, 2:]
            if len(temp_data) < self.window_size:
                data_interpolated = []
                for column in range(temp_data.shape[1]):
                    x_old = np.linspace(0, self.window_size - 1, len(temp_data))
                    x_new = np.linspace(0, self.window_size - 1, self.window_size)
                    spl = interpolate.splrep(x_old, temp_data[:, column])
                    interpolated_col = interpolate.splev(x_new, spl)
                    data_interpolated.append(interpolated_col.tolist())
                data_interpolated = np.array(data_interpolated)
                rul = self.data_rul[run - 1]
                data_interpolated = np.transpose(data_interpolated).tolist()
                x.append(data_interpolated)
                y.append(min(rul, self.max_rul))
                self.run_id.append(run)
            else:
                frame = temp_data[-self.window_size:, :]
                rul = self.data_rul[run - 1]
                x.append(frame)
                y.append(min(rul, self.max_rul))
                self.run_id.append(run)
        x = np.array(x)
        y = np.array(y) / self.max_rul
        self.run_id = np.array(self.run_id)

        if self.handcrafted:
            for i in range(len(x)):
                one_sample = x[i]
                hc.append(self.fea_extract(one_sample))
            mu = np.mean(hc, axis=0)
            sigma = np.std(hc, axis=0)
            eps = 1e-10
            hc = (hc - mu) / (sigma + eps)
            hc = np.array(hc)

            return x, hc, y
        else:
            return x, y

    @staticmethod
    def fea_extract(data):
        """
        Extracts mean and regression coefficient
        for feature
        :param data: list of size (Window_Size x 1)
        :return: list of size (2 x 1)
        """
        fea = []
        x = np.array(range(data.shape[0]))
        for i in range(data.shape[1]):
            fea.append(np.mean(data[:, i]))
            fea.append(np.polyfit(x.flatten(), data[:, i].flatten(), deg=1)[0])
        return fea

    def norm_data(self):
        """
        Performs standardization or normalization of
        data, depending on the flag self.standardize
        :return: numpy array
        """
        eps = 1e-12
        columns_to_keep = self.data[:, [0, 1]]
        columns_to_transform = self.data[:, 2:]
        if self.standardize:
            'If standardized data is needed: x = (x - mean(x))/sigma(x)'
            mu = np.mean(columns_to_transform, axis=0)
            sigma = np.std(columns_to_transform, axis=0)
            columns_to_transform = (columns_to_transform - mu) / (sigma + eps)
        else:
            'Simple normalization: x = (x - min(x))/(max(x) - min(x))'
            rows_min = np.min(columns_to_transform, axis=0)
            rows_max = np.max(columns_to_transform, axis=0)
            columns_to_transform = (columns_to_transform - rows_min) / (rows_max - rows_min + eps)

        transformed_data = np.concatenate((columns_to_keep.astype(int), columns_to_transform), axis=1)
        return transformed_data

    def get_full_run(self, run_id):
        """
        Returns X, y for specific run_id
        run_id starts with 1
        :param run_id: int [1, Number of runs]
        :return: tuple of numpy arrays
        """
        mask = np.asarray(self.run_id == run_id).nonzero()
        x_tensor = torch.from_numpy(self.x[mask]).to(torch.float32)
        y_tensor = torch.from_numpy(self.y[mask]).to(torch.float32)
        if self.handcrafted:
            hc_tensor = torch.from_numpy(self.hc[mask]).to(torch.float32)
            return x_tensor, hc_tensor, y_tensor
        return x_tensor, y_tensor

    def __getitem__(self, index):
        """
        Return
        :param index:
        :return: tuple of numpy arrays
        """
        x_tensor = torch.from_numpy(self.x[index]).to(torch.float32)
        y_tensor = torch.Tensor([self.y[index]]).to(torch.float32)
        if self.handcrafted:
            hc_tensor = torch.from_numpy(self.hc[index]).to(torch.float32)
            return x_tensor, hc_tensor, y_tensor
        return x_tensor, y_tensor

    def __len__(self):
        """
        Calculates the length of dataset
        :return: int
        """
        return len(self.y)


if __name__ == "__main__":
    dataset = CMAPSSSlidingWin(
                 "CMAPSSData/train_FD001.txt",
                 mode='train',
                 rul_path=None,
                 max_rul=150,
                 window_size=30,
                 standardize=False,
                 handcrafted=False,
                 drop_features=[5, 9, 10, 14, 20, 22, 23]
                 )
    print("run_id shape:", len(dataset.run_id))
    print("x shape:", dataset.x.shape)
    x, y = dataset.get_full_run(2)
    print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))

    dataset = CMAPSSSlidingWin(
        "CMAPSSData/train_FD001.txt",
        mode='train',
        rul_path=None,
        max_rul=150,
        window_size=30,
        standardize=False,
        handcrafted=True,
        drop_features=[5, 9, 10, 14, 20, 22, 23]
    )
    print("run_id shape:", len(dataset.run_id))
    print("x shape:", dataset.x.shape)
    x, hc, y = dataset.get_full_run(2)
    print("x shape:", tuple(x.shape), "hc shape:", tuple(hc.shape), "y shape:", tuple(y.shape))
