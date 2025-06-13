import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
import torch


class CSVScaledDataset(Dataset):
  def __init__(self, path, seq_len, split_idx_lst=None):
    data_array = pd.read_csv(path)
    self.features =  data_array.columns.tolist()
    self.seq_len = seq_len
    self.raw_data = torch.tensor(data_array.values, dtype=torch.float)

    # Calculate min and max for min-max scaling
    self.min = self.raw_data.min(dim=0, keepdim=True).values
    self.max = self.raw_data.max(dim=0, keepdim=True).values


    if split_idx_lst is None:  # Use full dataset if no split index provided
        self.idx_lst = np.arange(len(self.raw_data) - seq_len)
    else:
        self.idx_lst = split_idx_lst

  def __len__(self):
    return len(self.idx_lst)

  def __getitem__(self, idx):
    target_idx = self.idx_lst[idx]
    sample = self.raw_data[target_idx:target_idx+self.seq_len]
    # Apply min-max scaling
    sample = (sample - self.min) / (self.max - self.min + 1e-7)
    return sample # (B, L, K)
    
class NPYDataset(Dataset):
  def __init__(self, path, seq_len, split_idx_lst=None):
    data_array = np.load(path)
    self.data = torch.tensor(data_array, dtype=torch.float)
    self.seq_len = seq_len
    if split_idx_lst is None:  # Use full dataset if no split index provided
        self.idx_lst = np.arange(len(self.data) - seq_len)
    else:
        self.idx_lst = split_idx_lst

  def __len__(self):
    return len(self.idx_lst)

  def __getitem__(self, idx):
    target_idx = self.idx_lst[idx]
    sample = self.data[target_idx:target_idx+self.seq_len]

    return sample # (B, L, K)
    



def get_csv_scaled_dataloader(dataset_name, seq_len=24, batch_size=16):
    
  path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/'
    ) + dataset_name + '.csv'

  full_dataset = CSVScaledDataset(path, seq_len)
  data_range = len(full_dataset)
  train_dataset = CSVScaledDataset(path, seq_len, [i for i in range (int(data_range * 0.9))])
  test_dataset = CSVScaledDataset(path, seq_len, [i for i in range (int(data_range * 0.9), data_range)])
  dataset_info = {
    'features': full_dataset.features,
    'max': full_dataset.min,
    'min': full_dataset.max,
    'nsample': len(full_dataset)
  }
  full_loader = DataLoader(full_dataset, batch_size, shuffle=1)
  train_loader = DataLoader(train_dataset, batch_size, shuffle=1)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=1)

  return full_loader, train_loader, test_loader, dataset_info

