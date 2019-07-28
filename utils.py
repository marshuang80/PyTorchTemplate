from torch.utils.data import DataLoader
from dataset import Dataset

def get_dataloader(data_dir, batch_size, shuffle=False):
    params = {'batch_size': batch_size,              
              'shuffle': shuffle}                    
    dataset = Dataset(data_dir)                    
    dataloader = DataLoader(dataset, **params)

    return dataloader

