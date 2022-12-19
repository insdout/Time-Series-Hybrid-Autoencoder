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
                 drop_features=True
                 ):
        self.data = np.loadtxt(fname=data_path, dtype=np.float32)
        if drop_features:
            self.data = np.delete(self.data, [5, 9, 10, 14, 20, 22, 23], axis=1)
        if rul_path:
            self.data_rul = np.loadtxt(fname=rul_path, dtype=np.float32)
        self.mode = mode
        self.max_rul = max_rul
        self.window_size = window_size
        self.num_runs = int(self.data[-1, 0])
        self.standardize = standardize
        self.handcrafted = handcrafted

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
        x = np.array(x)
        y = np.array(y) / self.max_rul

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

            else:
                frame = temp_data[-self.window_size:, :]
                rul = self.data_rul[run - 1]
                x.append(frame)
                y.append(min(rul, self.max_rul))
        x = np.array(x)
        y = np.array(y) / self.max_rul

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
        fea = []
        x = np.array(range(data.shape[0]))
        for i in range(data.shape[1]):
            fea.append(np.mean(data[:, i]))
            fea.append(np.polyfit(x.flatten(), data[:, i].flatten(), deg=1)[0])
        return fea

    def norm_data(self):
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

    def __getitem__(self, index):
        x_tensor = torch.from_numpy(self.x[index]).to(torch.float32)
        y_tensor = torch.Tensor([self.y[index]]).to(torch.float32)
        if self.handcrafted:
            hc_tensor = torch.from_numpy(self.hc[index]).to(torch.float32)
            return x_tensor, hc_tensor, y_tensor
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y)
