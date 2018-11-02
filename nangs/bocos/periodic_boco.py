import numpy as np
from .boco import Boco
import torch
from torch.utils.data import DataLoader, Dataset
from ..vars import IndependentVar
from .loss import MSELoss 

class PeriodicBoco(Boco):
    def __init__(self, inputs1, inputs2):
        super().__init__()
        self.type = 'periodic'
        if not isinstance(inputs1, dict) or not isinstance(inputs2, dict): 
            raise ValueError('Inputs not valid')
        
        self.inputs1 = []
        for key in inputs1:
            self.inputs1 += [IndependentVar(key, inputs1[key])]
  
        self.inputs2 = []
        for key in inputs2:
            self.inputs2 += [IndependentVar(key, inputs2[key])]


    def check(self, inputs, outputs, params):
        # check that the inputs are correct
        for var in self.inputs1: assert var.name in inputs, '{} is not an input'.format(key)
        for var in self.inputs2: assert var.name in inputs, '{} is not an input'.format(key)

    def summary(self):
        print('Periodic Boco Summary:')
        print('Input 1: ', {var.name: var.values for var in self.inputs1})
        print('Input 2: ', {var.name: var.values for var in self.inputs2})
        print('')

    def initialize(self):
        self.dataset = PeriodicBocoDataset(self.inputs1, self.inputs2)
        self.dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=False, num_workers=4)
        self.loss = MSELoss()

    def computeLoss(self, model):
        loss = []
        for inputs1, inputs2 in self.dataloader:
            outputs1 = model(inputs1)
            outputs2 = model(inputs2)
            #print(inputs1, outputs1)
            #print(inputs2, outputs2)
            loss.append(self.loss(outputs1, outputs2))
        return np.array(loss).sum()

class PeriodicBocoDataset(Dataset):
    def __init__(self, inputs1, inputs2):
        self.inputs1, self.inputs2 = inputs1, inputs2 
        # length of the dataset (all possible combinations)
        self.len = 1
        for input in self.inputs1:
            self.len *= len(input.values)
        # modules
        self.mods = []
        for i, _ in enumerate(self.inputs1):
            mod = 1
            for j, input in enumerate(self.inputs1):
                if j < i:
                    mod *= len(input.values)
            self.mods.append(mod)  
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item1, item2 = np.zeros(len(self.inputs1)), np.zeros(len(self.inputs2))
        for i, input in enumerate(self.inputs1):
            item1[i] = input.values[(idx // self.mods[i]) % len(input.values)]
        for i, input in enumerate(self.inputs2):
            item2[i] = input.values[(idx // self.mods[i]) % len(input.values)]
        return torch.from_numpy(item1.astype(np.float32)), torch.from_numpy(item2.astype(np.float32))



