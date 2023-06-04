import glob
import h5py
import numpy as np
import pandas as pd
from utils.sliding_window import sliding_window
from torch.utils.data import Dataset, DataLoader


class MESADataset(Dataset):
    def __init__(self, path, split='train'):
        self.split = split
        self.path = glob.glob(path + 'MESA/*.h5')[0]
        self.file = h5py.File(self.path, 'r')
        self.x = np.asarray(self.file.get('x_' + self.split)[:, :, 0])
        self.y = np.asarray(self.file.get('y_' + self.split))
        self.y[self.y != 0] = 1
        self.file.close()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NCLDataset(Dataset):
    def __init__(self, path):
        self.files = glob.glob(path + 'NCL/*.csv')
        self.X, self.y = self.get_data()

    def get_data(self):
        X, y = [], []
        for file in self.files:
            df = pd.read_csv(file)
            X.append(df.iloc[:, 0].values)
            y.append(df.iloc[:, 1].values)
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        y[y != 0] = 1
        X_windowed = sliding_window(X, 101, 1)
        y_windowed = [[i[50]] for i in sliding_window(y, 101, 1)]
        return np.asarray(X_windowed), np.asarray(y_windowed).squeeze()

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

           
class SleepDataLoader():
    def __init__(self, data_dir):
        self.train_dataset = MESADataset(data_dir, split='train')
        self.val_dataset = MESADataset(data_dir, split='val')
        self.test_dataset = MESADataset(data_dir, split='test')
        self.ncl_dataset = NCLDataset(data_dir)

    def train(self, batch_size=256):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
    
    def val(self, batch_size=256):
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
    
    def test(self, batch_size=256):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    def ncl(self, batch_size=256):
        return DataLoader(self.ncl_dataset, batch_size=batch_size, shuffle=False)



