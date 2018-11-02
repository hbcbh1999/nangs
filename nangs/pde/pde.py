import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np 

from ..vars import IndependentVar, DependentVar, Param
from .solution import Solution

class PDE:
    def __init__(self, inputs=None, outputs=None, params=None):
        self.inputs = []
        self.outputs = []
        self.params = []
        self.bocos = []
        self.addInputs(inputs)
        self.addOutputs(outputs)
        self.addParams(params)
        self.model = None
        self.optimizer = None
        self.init = False
        self.dataset = None
        self.epochs = 0
        self.bs = 0

    def addInputs(self, inputs):
        if inputs:
            if isinstance(inputs, str):
                for var in self.inputs:
                    if var.name == inputs: raise ValueError('input {} already exists'.format(var.name))
                for var in self.outputs:
                    if var.name == inputs: raise ValueError('input {} already exists'.format(var.name))
                for var in self.params:
                    if var.name == inputs: 
                        var.isInput = True
                        var.values = []
                self.inputs += [IndependentVar(inputs)]
            elif isinstance(inputs, list):
                for input in inputs:
                    if not isinstance(input, str): raise ValueError('input must be a string')
                    for var in self.inputs:
                        if var.name == input: raise ValueError('input {} already exists'.format(input))
                    for var in self.outputs:
                        if var.name == input: raise ValueError('input {} already exists'.format(input))
                    for var in self.params:
                        if var.name == inputs: 
                            var.isInput = True
                            var.values = []
                    self.inputs += [IndependentVar(input)]
            elif isinstance(inputs, dict):   
                for key in inputs:
                    addKey = True
                    for var in self.inputs:
                        if var.name == key: 
                            var.values = inputs[key]
                            addKey = False
                    for var in self.outputs:
                        if var.name == key: raise ValueError('input {} already exists'.format(var.name))
                    for var in self.params:
                        if var.name == key: 
                            var.isInput = True
                            var.values = inputs[key]
                    if addKey: self.inputs += [IndependentVar(key, inputs[key])]
            else:
               raise ValueError('Inputs are not valid')   

    def addOutputs(self, outputs):
        if outputs:
            if isinstance(outputs, str):
                for var in self.inputs:
                    if var.name == outputs: raise ValueError('output {} already exists'.format(var.name))
                for var in self.outputs:
                    if var.name == outputs: raise ValueError('output {} already exists'.format(var.name))
                for var in self.params:
                    if var.name == outputs: raise ValueError('output {} already exists'.format(var.name))
                self.outputs += [DependentVar(outputs)]
            elif isinstance(outputs, list):
                for output in outputs:
                    if not isinstance(output, str): raise ValueError('output must be a string')
                    for var in self.inputs:
                        if var.name == output: raise ValueError('output {} already exists'.format(output))
                    for var in self.outputs:
                        if var.name == output: raise ValueError('output {} already exists'.format(output))
                    for var in self.params:
                        if var.name == output: raise ValueError('output {} already exists'.format(output))
                    self.outputs += [DependentVar(output)]
            else:
               raise ValueError('Inputs are not valid')         
    
    def addParams(self, params):
        if params:
            if isinstance(params, str):
                isInput = False 
                values = []
                for var in self.inputs:
                    if var.name == params: 
                        isInput = True
                        values = var.values
                for var in self.params:
                    if var.name == params: raise ValueError('param {} already exists'.format(var.name))
                for var in self.outputs:
                    if var.name == params: raise ValueError('param {} already exists'.format(var.name))
                self.params += [Param(params, values, isInput)]
            elif isinstance(params, list):
                for param in params:
                    isInput = False 
                    values = []
                    for var in self.inputs:
                        if var.name == params: 
                            isInput = True
                            values = var.values
                    if not isinstance(param, str): raise ValueError('param must be a string')
                    for var in self.params:
                        if var.name == param: raise ValueError('param {} already exists'.format(param))
                    for var in self.outputs:
                        if var.name == param: raise ValueError('param {} already exists'.format(param))
                    self.params += [Param(param, values, isInput)]
            elif isinstance(params, dict):        
                for key in params:
                    isInput = False
                    addKey = True
                    values = params[key]
                    for var in self.inputs:
                        if var.name == key: 
                            isInput = True
                            values = var.values
                    for var in self.params:
                        if var.name == key: 
                            var.values = params[key]
                            addKey = False
                    for var in self.outputs:
                        if var.name == key: raise ValueError('param {} already exists'.format(var.name))
                    if addKey: self.params += [Param(key, values, isInput=isInput)]

    def addBoco(self, boco):
        boco.check(self.getVarNames(self.inputs), self.getVarNames(self.outputs), self.getVarNames(self.params))
        self.bocos += [boco]

    def getVarNames(self, vars):
        return [var.name for var in vars]

    def summary(self):
        print('inputs: ', {var.name: var.values for var in self.inputs})
        print('outputs: ', {var.name: var.values for var in self.outputs})
        print('params: ', {var.name: (var.isInput, var.values) for var in self.params})
        print('bocos: ', [boco.type for boco in self.bocos])
        print('')

    def bocoSummary(self):
        for boco in self.bocos: boco.summary() 
            
    def computePdeLoss(self):
        print('This function has to be overloaded by a child class!')

    def buildModel(self, topo):
        n_inputs, n_outputs = len(self.inputs), len(self.outputs)
        self.model = Solution(n_inputs, n_outputs, topo['layers'], topo['neurons'], topo['activations'])

    def setSolverParams(self, lr=0.01, epochs=10, batch_size=10):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.bs = batch_size
        for boco in self.bocos:
            boco.setSolverParams(batch_size)

    def initialize(self):
        self.dataset = PDEDataset(self.inputs)
        self.dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=False, num_workers=4)
        for boco in self.bocos:
            boco.initialize()
        self.init = True

    def solve(self, each_epoch_print=100):
        if not self.init: self.initialize()
        model = self.model 
        optimizer = self.optimizer
        params = {param.name: torch.FloatTensor(param.values) for param in self.params}
        for epoch in range(self.epochs):
            model.train()
            total_loss = []

            for inputs in self.dataloader:
                # compute pde solution
                inputs = Variable(inputs, requires_grad=True)
                outputs = model(inputs)
                # compute gradients of outputs w.r.t. inputs
                grads = self.computeGrads(inputs, outputs)
                # compute loss
                loss = self.computePdeLoss(grads, params).pow(2).mean()
                # compute bocos loss
                for boco in self.bocos:
                    loss += boco.computeLoss(model)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.append(loss.data)

            if not epoch % each_epoch_print or (epoch+1) == self.epochs:
                print('Epoch: {}, Loss: {:4f} '.format(epoch, np.array(total_loss).mean()))

    def computeGrads(self, inputs, outputs):
        # compute gradients
        _grads, = torch.autograd.grad(outputs, inputs, 
                    grad_outputs=outputs.data.new(outputs.shape).fill_(1),
                    create_graph=True, only_inputs=True)
        # assign keys
        grads = {}
        for i, output in enumerate(self.outputs):
            grads[output.name] = {}
            for j, input in enumerate(self.inputs):
                grads[output.name][input.name] = _grads[:,j] # ???
        return grads

    def eval(self, inputs):
        # set values of inpenedent vars 
        for key in inputs: 
            for var in self.inputs:
                if var.name == key: var.values = inputs[key]  
        # build dataset
        dataset = PDEDataset(self.inputs)
        outputs = []
        self.model.eval()
        for i in range(len(dataset)):
            input = dataset[i]
            outputs.append(self.model(input).detach().numpy())
        return outputs

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
