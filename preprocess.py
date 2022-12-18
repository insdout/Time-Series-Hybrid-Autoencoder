import torch
from torch.utils.data import Dataset
from scipy import interpolate
import numpy as np


class CMAPSSDataset(Dataset):
    def __init__(self, data_path, mode='train', rul_path=None, max_rul=150, window_size=30):
        self.data = np.loadtxt(fname=data_path, dtype=np.float32)
        if rul_path:
            self.data_rul = np.loadtxt(fname=rul_path, dtype=np.float32)
        self.mode = mode
        self.max_rul = max_rul
        self.window_size = window_size
        self.num_runs = int(self.data[-1, 0])

        if mode == 'train':
            self.x, self.y = self.prepare_train()
        elif mode == 'test':
            self.x, self.y = self.prepare_test()

    def prepare_train(self):
        data = self.norm_data()
        x = []
        y = []
        for run in range(1, self.num_runs + 1):
            mask = np.asarray(data[:, 0] == run).nonzero()[0]
            temp_data = data[mask, 2:]

            for i in range(len(temp_data) - self.window_size + 1):
                frame = temp_data[i:i + self.window_size, 2:]
                rul = len(temp_data) - i - self.window_size
                x.append(frame)
                y.append(min(rul, self.max_rul))
        x = np.array(x)
        y = np.array(y) / self.max_rul
        return x, y

    def prepare_test(self):
        data = self.norm_data()
        x = []
        y = []
        for run in range(1, self.num_runs + 1):
            mask = np.asarray(data[:, 0] == run).nonzero()[0]
            temp_data = data[mask, 2:]
            if len(temp_data) < self.window_size:
                data_interpolated = []
                for column in range(2, temp_data.shape[1]):
                    x_old = np.linspace(0, self.window_size - 1, len(temp_data))
                    x_new = np.linspace(0, self.window_size - 1, self.window_size)
                    spl = interpolate.splrep(x_old, temp_data[:, column])
                    interpolated_col = interpolate.splev(x_new, spl)
                    data_interpolated.append(interpolated_col.tolist())
                data_interpolated = np.array(data_interpolated)
                # print(len(data_interpolated), len(data_interpolated[0]))
                rul = self.data_rul[run - 1]
                data_interpolated = np.transpose(data_interpolated).tolist()
                # print()
                # print('inter', len(data_interpolated), len(data_interpolated[0]))
                x.append(data_interpolated)
                y.append(min(rul, self.max_rul))
                # print("interpolated")
                # print("run:", run)
                # print("interpolated:",np.transpose(data_interpolated).shape)
                # print(min(rul, self.max_rul))


            else:
                for i in range(len(temp_data) - self.window_size + 1):
                    frame = temp_data[i:i + self.window_size, 2:]
                    rul = self.data_rul[run - 1] + len(temp_data) - i - self.window_size
                    # print("frame", len(frame), len(frame[0]))
                    x.append(frame)
                    y.append(min(rul, self.max_rul))
        x = np.array(x)
        y = np.array(y) / self.max_rul
        return x, y

    def norm_data(self):
        columns_to_keep = self.data[:, [0, 1]]
        columns_to_transform = self.data[:, 2:]

        eps = 1e-12
        rows_min = np.min(columns_to_transform, axis=0)
        rows_max = np.max(columns_to_transform, axis=0)

        columns_to_transform = (columns_to_transform - rows_min) / (rows_max - rows_min + eps)
        transformed_data = np.concatenate((columns_to_keep.astype(int), columns_to_transform), axis=1)
        return transformed_data

    def __getitem__(self, index):
        x_tensor = torch.from_numpy(self.x[index]).to(torch.float32)
        y_tensor = torch.Tensor([self.y[index]]).to(torch.float32)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y)

