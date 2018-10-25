from .boco import Boco

class DirichletBoco(Boco):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.type = 'dirichlet'
        if not isinstance(inputs, dict) or not isinstance(outputs, dict): 
            raise ValueError('Inputs not valid')
        for key in inputs:
            if isinstance(inputs[key], int) or isinstance(inputs[key], float): inputs[key] = [inputs[key]]
        for key in outputs:
            if isinstance(outputs[key], int) or isinstance(outputs[key], float): outputs[key] = [outputs[key]]
        self.inputs = inputs
        self.outputs = outputs

    def check(self, inputs, outputs, params):
        # check that the inputs are correct
        for key in self.inputs: assert key in inputs, '{} is not an input'.format(key)
        for key in self.outputs: assert key in outputs, '{} is not an output'.format(key)

    def summary(self):
        print('Dirichlet Boco Summary:')
        print('Inputs: ', self.inputs)
        print('Outputs: ', self.outputs)
        print('')