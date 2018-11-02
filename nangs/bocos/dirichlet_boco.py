import numpy as np
from .boco import Boco
import torch
from torch.utils.data import DataLoader, Dataset
from ..vars import IndependentVar, DependentVar
from .loss import MSELoss 

class DirichletBoco(Boco):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.type = 'dirichlet'
        if not isinstance(inputs, dict) or not isinstance(outputs, dict): 
            raise ValueError('Inputs not valid')
        
        self.inputs = []
        for key in inputs:
            self.inputs += [IndependentVar(key, inputs[key])]
  
        self.outputs = []
        for key in outputs:
            self.outputs += [DependentVar(key, outputs[key])]

    def check(self, inputs, outputs, params):
        # check that the inputs are correct
        for var in self.inputs: assert var.name in inputs, '{} is not an input'.format(key)
        for var in self.outputs: assert var.name in outputs, '{} is not an output'.format(key)

    def summary(self):
        print('Dirichlet Boco Summary:')
        print('Inputs: ', {var.name: var.values for var in self.inputs})
        print('Outpus: ', {var.name: var.values for var in self.outputs})
        print('')
    
    def initialize(self):
        self.dataset = DirichletBocoDataset(self.inputs, self.outputs)
        self.dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=False, num_workers=4)
        self.loss = MSELoss()

    def computeLoss(self, model):
        loss = []
        for inputs, outputs in self.dataloader:
            preds = model(inputs)
            #print(inputs, preds, outputs)
            loss.append(self.loss(preds, outputs))
        return np.array(loss).sum()

class DirichletBocoDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs, self.outputs = inputs, outputs
        # length of the dataset (all possible combinations)
        self.len = 1
        for input in self.inputs:
            self.len *= len(input.values)
        # modules
        self.mods = []
        for i, _ in enumerate(self.inputs):
            mod = 1
            for j, input in enumerate(self.inputs):
                if j < i:
                    mod *= len(input.values)
            self.mods.append(mod)  
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item1, item2 = np.zeros(len(self.inputs)), np.zeros(len(self.outputs))
        for i, input in enumerate(self.inputs):
            item1[i] = input.values[(idx // self.mods[i]) % len(input.values)]
        for i, output in enumerate(self.outputs):
            item2[i] = output.values[(idx // self.mods[i]) % len(output.values)]
            #print(self.outputs[0].values[(idx // self.mods[i]) % len(output.values)])
        return torch.from_numpy(item1.astype(np.float32)), torch.from_numpy(item2.astype(np.float32))
