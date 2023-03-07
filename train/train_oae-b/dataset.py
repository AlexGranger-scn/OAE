from torch.utils.data import DataLoader,Dataset
import os
import pandas as pd
from scipy.io import wavfile
import numpy as np
import torch
import pdb


class CustomDataset(Dataset):
    def __init__(self, data, emb_dict):
        self.emb = emb_dict
        with open(data, 'r') as f:
            self.files = f.readlines() # 得到文件名及标签
    def __len__(self): 
        return len(self.files) - 1 #去掉表头那一行
    def __getitem__(self, idx):
        #print(self.files[idx].strip().split(',')[3])
        data_id = self.files[idx].strip().split(',')[1]
        emb_data = self.emb[self.files[idx].strip().split(',')[2]]
        label = int(self.files[idx].strip().split(',')[3])
        #path = os.path.join(self.root,file_path)
        return torch.from_numpy(emb_data)


