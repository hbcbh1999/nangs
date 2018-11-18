import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from ..utils import *
from .solution import Solution
from .dataset import PDEDataset

class PDE:
    def __init__(self, inputs=None, outputs=None, params=[]):

        # assert vars are lists of strings
        checkIsListOfStr(inputs, 'Inputs')
        checkIsListOfStr(outputs, 'Outputs')
        checkIsListOfStr(params, 'Params')

        # assert no repeated keys
        checkNoRepeatedSame(inputs)
        checkNoRepeatedSame(outputs)
        checkNoRepeatedSame(params)
        checkNoRepeated(inputs, outputs)
        checkNoRepeated(inputs, params)
        checkNoRepeated(params, outputs)

        self.inputs = inputs
        self.outputs = outputs
        self.params = params

        # initialize values
        self.inputValues = initValues(self.inputs)
        self.outputValues = initValues(self.outputs)
        self.paramValues = initValues(self.params)

        self.bocos = []

        self.model = None
        self.optimizer = None
        self.init = False
        self.dataset = None
        self.epochs = 0
        self.bs = 0

    def setValues(self, values):
        checkValidDict(values)
        for key in values:
            value = values[key]
            if key in self.inputs: 
                setValue(self.inputs, self.inputValues, key, value)
                if key in self.params: 
                    setValue(self.params, self.paramValues, key, value)
            elif key in self.params: 
                setValue(self.params, self.paramValues, key, value)    
            elif key in self.outputs:
                raise ValueError('You cannot set values to outputs !')
            else:
                raise ValueError('Key '+key+' not found !')

    def addBoco(self, boco):
        self.bocos += [boco]

    def summary(self):
        print('inputs: ', {name: values for name, values in zip(self.inputs, self.inputValues)})
        print('outputs: ', {name: values for name, values in zip(self.outputs, self.outputValues)})
        print('params: ', {name: values for name, values in zip(self.params, self.paramValues)})
        print('bocos: ', [boco.type for boco in self.bocos])
        print('')

    def bocoSummary(self):
        for boco in self.bocos: boco.summary(self.inputs, self.outputs, self.params) 
            
    def computePdeLoss(self):
        print('This function has to be overloaded by a child class!')

    def buildModel(self, topo, device):
        n_inputs, n_outputs = len(self.inputs), len(self.outputs)
        self.model = Solution(n_inputs, n_outputs, topo['layers'], topo['neurons'], topo['activations'])
        self.model.to(device)

    def setSolverParams(self, lr=0.01, epochs=10, batch_size=10):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.bs = batch_size
        for boco in self.bocos:
            boco.setSolverParams(batch_size)

    def initialize(self):
        self.dataset = PDEDataset(self.inputValues)
        self.dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=True, num_workers=4)
        for boco in self.bocos:
            boco.initialize()
        self.init = True

    def solve(self, device, path, early_stop = 20):
        if not self.init: self.initialize()
        model = self.model 
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, min_lr=0.00001)
        params = {k: torch.FloatTensor(v).to(device) for k, v in zip(self.params, self.paramValues)}
        best_loss, stop = 1e8, 0
        for epoch in range(self.epochs):
            model.train()
            total_loss = []

            for inputs in tqdm(self.dataloader, ascii=True):
                # compute pde solution
                inputs = Variable(inputs, requires_grad=True)
                inputs = inputs.to(device)
                outputs = model(inputs)
                # compute gradients of outputs w.r.t. inputs
                grads, _inputs = self.computeGrads(inputs, outputs)
                # compute loss
                loss = self.computePdeLoss(grads, _inputs, params).pow(2).mean()
                # compute bocos loss
                for boco in self.bocos:
                    loss += boco.computeLoss(model, device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.append(loss.data)
                
            total_loss = np.array(total_loss).mean()
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.model.state_dict(), path)
                print('Best model, saved !')
                stop = 0

            stop += 1
            if stop > early_stop:
                print('Early stop !')
                break
            
            scheduler.step(total_loss)

            print('Epoch: {}, Loss: {:4f} '.format(epoch, total_loss))

    def computeGrads(self, inputs, outputs):
        # compute gradients
        _grads, = torch.autograd.grad(outputs, inputs, 
                    grad_outputs=outputs.data.new(outputs.shape).fill_(1),
                    create_graph=True, only_inputs=True)
        # assign keys to gradients
        grads = {}
        for output in self.outputs:
            grads[output] = {}
            for j, input in enumerate(self.inputs):
                grads[output][input] = _grads[:,j] # ???
        # assign keys to inputs
        _inputs = {}
        for i, input in enumerate(self.inputs):
            _inputs[input] = inputs[:,i]
        return grads, _inputs

    def eval(self, inputs, device):
        checkValidDict(inputs)
        checkDictArray(inputs, self.inputs)
        # set values of inpenedent vars 
        for key in inputs: 
            if key in self.inputs: 
                ix = self.inputs.index(key)
                self.inputValues[ix] = inputs[key] 
            else: 
                raise ValueError(key + ' is not an input')
        # build dataset
        dataset = PDEDataset(self.inputValues)
        outputs = []
        self.model.eval()
        for i in range(len(dataset)):
            input = dataset[i]
            input = input.to(device)
            outputs.append(self.model(input).cpu().detach().numpy())
        return outputs


