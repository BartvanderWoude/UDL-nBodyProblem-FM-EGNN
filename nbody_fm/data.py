import pandas as pd
import torch

from torch.utils.data import Dataset


class NBodyData(Dataset):
    def __init__(self, datafile, look_ahead=1):
        self.data = pd.read_csv(datafile, header=None)
        self.look_ahead = look_ahead
        self.length = len(self.data) - look_ahead

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x1_coors = torch.tensor(self.data.iloc[idx, 0:2].values, dtype=torch.float32)
        x2_coors = torch.tensor(self.data.iloc[idx, 2:4].values, dtype=torch.float32)
        x3_coors = torch.tensor(self.data.iloc[idx, 4:6].values, dtype=torch.float32)
        x1_vel = torch.tensor(self.data.iloc[idx, 6:8].values, dtype=torch.float32)
        x2_vel = torch.tensor(self.data.iloc[idx, 8:10].values, dtype=torch.float32)
        x3_vel = torch.tensor(self.data.iloc[idx, 10:12].values, dtype=torch.float32)

        x_coors = torch.stack([x1_coors, x2_coors, x3_coors])
        x_vel = torch.stack([x1_vel, x2_vel, x3_vel])

        y1_coors = torch.tensor(self.data.iloc[idx + self.look_ahead, 0:2].values, dtype=torch.float32)
        y2_coors = torch.tensor(self.data.iloc[idx + self.look_ahead, 2:4].values, dtype=torch.float32)
        y3_coors = torch.tensor(self.data.iloc[idx + self.look_ahead, 4:6].values, dtype=torch.float32)
        y1_vel = torch.tensor(self.data.iloc[idx + self.look_ahead, 6:8].values, dtype=torch.float32)
        y2_vel = torch.tensor(self.data.iloc[idx + self.look_ahead, 8:10].values, dtype=torch.float32)
        y3_vel = torch.tensor(self.data.iloc[idx + self.look_ahead, 10:12].values, dtype=torch.float32)

        y_coors = torch.stack([y1_coors, y2_coors, y3_coors])
        y_vel = torch.stack([y1_vel, y2_vel, y3_vel])

        return x_coors, x_vel, y_coors, y_vel
