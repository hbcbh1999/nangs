import numpy as np
from .boco import Boco
import torch
from torch.utils.data import DataLoader, Dataset
from .loss import MSELoss 
from ..utils import *

class DirichletBoco(Boco):
    def __init__(self, inputs, outputs, boco_inputs, boco_outputs):
        super().__init__()
        self.type = 'dirichlet'

        checkValidDict(boco_inputs)
        checkValidDict(boco_outputs)

        # check that the length of the inputs is the same than the outputs

        # check that all inputs and outputs are present
        checkDictArray(inputs, boco_inputs)
        checkDictArray(outputs, boco_outputs)

        # create empty list with same dimensions that inputs in pde
        self.inputs, self.outputs = [], []
        for input in inputs:
            self.inputs.append([])
        for output in outputs:
            self.outputs.append([])

        # extract arrays from dict and store in list, ordered by inputs in the pde
        for k in inputs:
            ix = inputs.index(k)
            self.inputs[ix] = boco_inputs[k]

        for k in outputs:
            ix = outputs.index(k)
            self.outputs[ix] = boco_outputs[k]

    def summary(self, inputs, outputs, params):
        print('Dirichlet Boco Summary:')
        print('Inputs: ', {name: values for name, values in zip(inputs, self.inputs)})
        print('Outputs: ', {name: values for name, values in zip(outputs, self.outputs)})
        print('')
    
    def initialize(self):
        self.dataset = DirichletBocoDataset(self.inputs, self.outputs)
        self.dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=False, num_workers=4)
        self.loss = MSELoss()

    def computeLoss(self, model):
        loss = []
        for inputs, outputs in self.dataloader:
            preds = model(inputs)
            loss.append(self.loss(preds, outputs))
        return np.array(loss).sum()

class DirichletBocoDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs, self.outputs = inputs, outputs
        # length of the dataset (all possible combinations)
        self.len = 1
        for input in self.inputs:
            self.len *= len(input)
        # modules
        self.mods = []
        for i, _ in enumerate(self.inputs):
            mod = 1
            for j, input in enumerate(self.inputs):
                if j < i:
                    mod *= len(input)
            self.mods.append(mod)  
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item1, item2 = np.zeros(len(self.inputs)), np.zeros(len(self.outputs))
        for i, input in enumerate(self.inputs):
            item1[i] = input[(idx // self.mods[i]) % len(input)]
        for i, output in enumerate(self.outputs):
            item2[i] = output[(idx // self.mods[i]) % len(output)]
        return torch.from_numpy(item1.astype(np.float32)), torch.from_numpy(item2.astype(np.float32))
