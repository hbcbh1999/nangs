import numpy as np
from .boco import Boco
import torch
from torch.utils.data import DataLoader, Dataset
from .loss import MSELoss 
from ..utils import *

class PeriodicBoco(Boco):
    def __init__(self, inputs, inputs1, inputs2):
        super().__init__()
        self.type = 'periodic'
        
        # check for dict with numpy arrays, same inputs and same length
        checkValidDict(inputs1)
        checkValidDict(inputs2)
        assert len(inputs1) == len(inputs2), 'Inputs must have same length !'
        for k in inputs1:
            assert k in inputs2, k+' must be present in both inputs !'
            assert len(inputs1[k]) == len(inputs2[k]), k+' must have same length in both inputs !'

        # check that all inputs are present
        checkDictArray(inputs, inputs1)
        checkDictArray(inputs, inputs2)

        # create empty list with same dimensions that inputs in pde
        self.inputs1, self.inputs2 = [], []
        for input in inputs:
            self.inputs1.append([])
            self.inputs2.append([])

        # extract arrays from dict and store in list, ordered by inputs in the pde
        for k in inputs1:
            ix = inputs.index(k)
            self.inputs1[ix] = inputs1[k]

        for k in inputs2:
            ix = inputs.index(k)
            self.inputs2[ix] = inputs2[k]

    def summary(self, inputs, outputs, params):
        print('Periodic Boco Summary:')
        print('Input 1: ', {name: values for name, values in zip(inputs, self.inputs1)})
        print('Input 2: ', {name: values for name, values in zip(inputs, self.inputs2)})
        print('')

    def initialize(self):
        self.dataset = PeriodicBocoDataset(self.inputs1, self.inputs2)
        self.dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=True, num_workers=4)
        self.loss = MSELoss()

    def computeLoss(self, model, device):
        loss = []
        for inputs1, inputs2 in self.dataloader:
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            outputs1 = model(inputs1)
            outputs2 = model(inputs2)
            loss.append(self.loss(outputs1, outputs2))
        return np.array(loss).sum()

class PeriodicBocoDataset(Dataset):
    def __init__(self, inputs1, inputs2):
        self.inputs1, self.inputs2 = inputs1, inputs2 
        # length of the dataset (all possible combinations)
        self.len = 1
        for input in self.inputs1:
            self.len *= len(input)
        # modules
        self.mods = []
        for i, _ in enumerate(self.inputs1):
            mod = 1
            for j, input in enumerate(self.inputs1):
                if j < i:
                    mod *= len(input)
            self.mods.append(mod)  
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item1, item2 = np.zeros(len(self.inputs1)), np.zeros(len(self.inputs2))
        for i, input in enumerate(self.inputs1):
            item1[i] = input[(idx // self.mods[i]) % len(input)]
        for i, input in enumerate(self.inputs2):
            item2[i] = input[(idx // self.mods[i]) % len(input)]
        return torch.from_numpy(item1.astype(np.float32)), torch.from_numpy(item2.astype(np.float32))



