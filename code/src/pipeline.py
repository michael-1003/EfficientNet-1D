import os
import random

import torch
from torch.utils.data import Dataset

import numpy as np


#%%
class NLUDataset(Dataset):
    def __init__(self, data_path, mode='train', normalization=False, random_seed=42):
        print('Construct %s data ...... ' % mode, end='')
        self.normalization = normalization
        # Load data --------------------------------
        A = np.genfromtxt(data_path,delimiter=',')
        if A[0,0] != A[0,0]: # Check [0,0] if it is nan
            with open(data_path, 'r') as f:
                line = f.readline()
            A00 = float(line[1:].split(',')[0])
            A[0,0] = A00
        # Get information about label --------------------------------
        self.labels = np.unique(A[:,-1].astype(int))
        num_datas = []
        for label in self.labels:
            num_datas.append(np.sum(A[:,-1].astype(int) == label))
        # Select index to separate train/valid/test --------------------------------
        train_ids = []
        valid_ids = []
        test_ids = []
        start_ind = 0
        for num_data in num_datas:
            train_id, valid_id, test_id = self.random_index_sampler(start_ind, num_data, random_seed)
            train_ids += train_id
            valid_ids += valid_id
            test_ids += test_id
            start_ind += num_data
        # We got index to select. So reset the random seed
        random.seed()
        # Select index by mode
        if mode == 'train': index = train_ids
        elif mode == 'valid': index = valid_ids
        elif mode == 'test': index = test_ids
        else: raise(ValueError('There is no such mode!'))
        # Select data by index
        self.X = A[index,:-1]
        self.Label = A[index,-1].astype(int)
        print('Complete!')

    def random_index_sampler(self, start_index, num_data, random_seed):
        random.seed(random_seed)
        n_train = int(num_data * 0.6)
        n_valid = int(num_data * 0.2)
        n_test = num_data - n_train - n_valid
        indices = list(range(start_index, start_index+num_data))
        train_id = random.sample(indices, n_train)
        left0 = [x for x in indices if x not in train_id]
        valid_id = random.sample(left0, n_valid)
        left1 = [x for x in left0 if x not in valid_id]
        test_id = random.sample(left1, n_test)
        return train_id, valid_id, test_id

    def normalize(self, x):
        mu = np.mean(x)
        sigma = np.std(x)
        return (x-mu)/sigma

    def __len__(self):
        return(len(self.Label))

    def __getitem__(self, idx):
        if self.normalization: x = np.expand_dims(self.normalize(self.X[idx,:]), 0) # x become shape [1,L(=length)]
        else: x = np.expand_dims(self.X[idx,:], 0) # x become shape [1,L(=length)]
        label = self.Label[idx]
        return x, label


#%% Debug
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_path = '../data/NLU/d1.csv'
    nludataset = NLUDataset(data_path, mode='train', normalization=False)
    print(nludataset.__len__())
    print(nludataset.labels)
    print(nludataset[0])

    plt.plot(nludataset[0][0][0])
    plt.savefig('../temp/data.png')
    

    dataloader = DataLoader(nludataset, batch_size=16, shuffle=False, num_workers=4)
    for x, label in dataloader:
        print(x)
        print(x.shape)
        print(label)
        break