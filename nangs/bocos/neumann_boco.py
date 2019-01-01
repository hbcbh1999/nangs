import numpy as np
from .boco import Boco
import torch
from torch.utils.data import DataLoader, Dataset
from ..utils import *
from torch.autograd import Variable

class NeumannBoco(Boco):
    def __init__(self, inputs, outputs, boco_inputs, grads):
        super().__init__()
        self.type = 'neumann'
        self.grads = grads

        checkValidDict(boco_inputs)

        # check that the length of the inputs is the same than the outputs

        # check that all inputs are present
        checkDictArray(inputs, boco_inputs)

        # create empty list with same dimensions that inputs in pde
        self.inputs, self.outputs = [], []
        for input in inputs:
            self.inputs.append([])

            
        # extract arrays from dict and store in list, ordered by inputs in the pde
        for k in inputs:
            ix = inputs.index(k)
            self.inputs[ix] = boco_inputs[k]
        
        self.outputs = outputs
        self._inputs = inputs

    def summary(self, inputs, outputs, params):
        print('Neumann Boco Summary:')
        print('Inputs: ', {name: values for name, values in zip(inputs, self.inputs)})   
        print('Grads: ', self.grads)
        print('')
    
    def initialize(self):
        self.dataset = NeumannBocoDataset(self.inputs)
        self.dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=True, num_workers=4)

    def computeLoss(self, model, device):
        loss = []
        for inputs in self.dataloader:
            inputs = Variable(inputs, requires_grad=True)
            inputs = inputs.to(device)
            preds = model(inputs)
            # compute gradients
            _grads, = torch.autograd.grad(preds, inputs, 
                        grad_outputs=preds.data.new(preds.shape).fill_(1),
                        create_graph=True, only_inputs=True)
            # assign keys to gradients
            grads = {}
            for output in self.outputs:
                grads[output] = {}
                for j, input in enumerate(self._inputs):
                    grads[output][input] = _grads[:,j] # ???
            # compute loss for corresponding gradients
            for g in self.grads:
                loss.append(grads[g][self.grads[g]].pow(2).sum())
            
        return np.array(loss).sum()

class NeumannBocoDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
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
        item1= np.zeros(len(self.inputs))
        for i, input in enumerate(self.inputs):
            item1[i] = input[(idx // self.mods[i]) % len(input)]
        return torch.from_numpy(item1.astype(np.float32))
