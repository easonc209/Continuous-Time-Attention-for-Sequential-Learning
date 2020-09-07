###########################
# Continuous-Time Attention for Sequential Learning
# Authors: Yi-Hsiang Chen
###########################

import os
import numpy as np
import random

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms

import lib.utils as utils

class MELD(object):

    def __init__(self, root, download=False,
        reduce='average', seq_length = 50,
        n_samples = None, device = torch.device("cpu")):

        self.root = root
        self.reduce = reduce
        self.seq_length = seq_length

        if download:
            self.download()

        if not self._check_exists():
            #raise RuntimeError('Dataset not found. You can use download=True to download it')
            self.preprocess()
        
        if device == torch.device("cpu"):
            data = torch.load(os.path.join(self.processed_folder, self.data_file), map_location='cpu')
        else:
            data = torch.load(os.path.join(self.processed_folder, self.data_file))

        self.train_data = data['train_data']
        #self.dev_data = data['dev_data']
        self.test_data = data['test_data']
        if n_samples is not None:
            self.train_data = self.train_data[:n_samples]

    def download(self):
        pass
    
    def preprocess(self):
        pass
         
       
    def _check_exists(self):
        if not os.path.exists(
            os.path.join(self.processed_folder, 'data.pt')
        ):
            return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def data_file(self):
        return 'data.pt'

    def __getitem__(self, index):
        return self.train_data[index]

    def __len__(self):
        return len(self.train_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Max length: {}\n'.format(self.seq_length)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str

    
def meld_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train"):
    """
    Expects a batch of time series data in the form of (feature, time, label) where
        - feature is a (T, D) tensor containing observed values for D variables.
        - label is a list of labels for the current patient, if labels are available. Otherwise None.
        - time start is a 1-dimensional tensor containing T time values of observations.
        - time end is a 1-dimensional tensor containing T time values of observations.
    Returns:
        combined_feature: (M, T, D) tensor containing the observed values.
        combined_time: The union of all time observations.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][0].shape[1]
    N = 1 # number of labels

    combined_tt, inverse_indices = torch.unique(torch.cat([ex[3] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D+1]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), 1]).to(device)
    combined_labels = torch.zeros([len(batch), len(combined_tt), N]).to(device) + 7
    
    max_val = torch.max(combined_tt)
    for b, (feature, label, s_time, e_time) in enumerate(batch):
        s_time = s_time.to(device).reshape(-1,1) / max_val
        e_time = e_time.to(device)
        feature = feature.to(device)
        #mask = torch.ones_like(feature).to(device)
        label = label.to(device)

        indices = inverse_indices[offset:offset + len(e_time)]
        offset += len(e_time)
        combined_vals[b, indices.squeeze()] = torch.cat((feature,s_time),1)
        combined_mask[b, indices.squeeze()] = 1 #mask
        combined_labels[b, indices] = label.reshape(-1,1).float()

    combined_tt = combined_tt.float()

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "data": combined_vals, 
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels}

    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
    return data_dict