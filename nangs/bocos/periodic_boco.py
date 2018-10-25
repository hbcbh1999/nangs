from .boco import Boco

class PeriodicBoco(Boco):
    def __init__(self, inputs1, inputs2):
        super().__init__()
        self.type = 'periodic'
        if not isinstance(inputs1, dict) or not isinstance(inputs2, dict): 
            raise ValueError('Inputs not valid')
        for key in inputs1:
            if isinstance(inputs1[key], int) or isinstance(inputs1[key], float): inputs1[key] = [inputs1[key]]
        for key in inputs2:
            if isinstance(inputs2[key], int) or isinstance(inputs2[key], float): inputs2[key] = [inputs2[key]]
        self.inputs1 = inputs1
        self.inputs2 = inputs2

    def check(self, inputs, outputs, params):
        # check that the inputs are correct
        for key in self.inputs1: assert key in inputs, '{} is not an input'.format(key)
        for key in self.inputs2: assert key in inputs, '{} is not an input'.format(key)

    def summary(self):
        print('Periodic Boco Summary:')
        print('Input 1: ', self.inputs1)
        print('Input 2: ', self.inputs2)
        print('')