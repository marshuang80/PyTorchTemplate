from torch.utils.data import Dataset
import numpy as np
import pickle

class Dataset(Dataset):

    def __init__(self, data_path):

        self.data = None
        self.x = None
        self.y = None


    def __len__(self):

        return len(self.x)


    def __getitem__(self, idx):

        x = self.x[idx]
        y = self.y[idx]

        return x, y


