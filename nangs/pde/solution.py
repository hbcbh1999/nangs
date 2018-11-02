import torch
import torch.nn as nn

class Solution(nn.Module):
    def __init__(self, inputs, outputs, layers, neurons, activations):
        super(Solution, self).__init__()
        # checks
        assert isinstance(inputs, int), 'inputs must be a an integer'
        assert isinstance(outputs, int), 'outputs must be a an integer'
        assert isinstance(layers, int), 'layers must be a an integer'
        assert isinstance(neurons, int), 'neurons must be a an integer'
        assert isinstance(activations, str), 'activation must be a string'
        # activaton function
        self.activation = None
        if activations == 'relu': self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid': self.activation = nn.Sigmoid(inplace=True)
        else: raise ValueError('activation function {} not valid'.format(activation))
        # layers
        self.fc_in = nn.Sequential(nn.Linear(inputs, neurons), self.activation)
        self.fc_hidden = nn.ModuleList()
        for layer in range(layers):
            self.fc_hidden.append(nn.Sequential(nn.Linear(neurons, neurons), self.activation))
        self.fc_out = nn.Linear(neurons, outputs)

    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.fc_hidden:
            x = layer(x)
        x = self.fc_out(x)
        return x

        
