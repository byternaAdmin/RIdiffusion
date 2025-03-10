
from torch_geometric.data import Dataset, download_url,Batch,Data
import torch
import os
from torch_geometric.loader import DataListLoader, DataLoader
import random

class RNAsolo(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, baseDIR,transform=None, pre_transform=None, pre_filter=None,pred_sasa = False):
        super().__init__(baseDIR, transform, pre_transform, pre_filter)
        'Initialization'
        self.list_IDs = list_IDs
        self.baseDIR = baseDIR
        self.pred_sasa = pred_sasa

    def len(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        data = torch.load(self.baseDIR+ID)
        del data['distances']
        del data['edge_dist']
        mu_r_norm=data.mu_r_norm
        extra_x_feature = torch.cat([data.x[:,4:],mu_r_norm],dim = 1)
        graph = Data(
            x=data.x[:, :4],
            extra_x = extra_x_feature,
            pos=data.pos,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            ss = data.ss[:data.x.shape[0],:],
            # sasa = data.x[:,4]
        )
        return graph