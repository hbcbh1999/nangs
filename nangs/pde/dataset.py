from torch.utils.data import Dataset
import numpy as np 
import torch 

class PDEDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs 
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
        item = np.zeros(len(self.inputs))
        for i, input in enumerate(self.inputs):
            item[i] = input.values[(idx // self.mods[i]) % len(input.values)]
        return torch.from_numpy(item.astype(np.float32))