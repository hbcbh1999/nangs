from .utils import *

class PDE:
    def __init__(self, inputs=None, outputs=None, params=None):
        # check that inputs, outputs and params are arrays with strings
        self.inputs = checkInputs(inputs, 'Inputs')
        self.outputs = checkInputs(outputs, 'Outputs')
        self.params = checkInputs(params, 'Params')
        checkSelfRepeated(self.inputs)
        checkSelfRepeated(self.outputs)
        checkSelfRepeated(self.params)
        checkRepeated(self.inputs, self.outputs)
        checkRepeated(self.outputs, self.params)
        self.input_values = {}
        self.params_values = {}

    def addInputs(self, inputs=None):
        self.inputs = addInputs(inputs, 'Inputs', self.inputs, self.outputs)

    def addOutputs(self, outputs=None):
        self.outputs = addInputs(outputs, 'Outputs', self.outputs, self.inputs, self.params)

    def addParams(self, params=None):
        self.params = addInputs(params, 'Params', self.params, self.outputs)

    def setInputs(self, inputs=None):
        if not inputs: raise ValueError('Inputs are empty !')
        if not isinstance(inputs, dict): raise ValueError('Inputs must be a dict !')
        for key in inputs:
            if not key in self.inputs: raise ValueError('{} is not an input'.format(key))
            if not isinstance(inputs[key], list): raise ValueError('Each input must be a list !')
            # TODO: check all elements in list are numbers !
            #for input in inputs[key]:
            #    if not isinstance(input, int) or not isinstance(input, float): raise ValueError('Each input must be a list of numbers!')
            self.input_values[key] = inputs[key]

    def setParams(self, params=None):
        if not params: raise ValueError('Params are empty !')
        if not isinstance(params, dict): raise ValueError('Params must be a dict !')
        for key in params:
            if not key in self.params: raise ValueError('{} is not a param'.format(key))
            # only allow lists if param is input 
            if isinstance(params[key], list) and not key in self.inputs: raise ValueError('List params are only allowed if param is input !')
            if isinstance(params[key], list):
                # TODO: check all elements in list are numbers !
                self.params_values[key] = params[key]
            elif isinstance(params[key], int) or isinstance(params[key], float):
                self.params_values[key] = params[key]
            else: raise ValueError('Params must be numbers !')

        
    def summary(self):
        print('inputs: ', self.inputs)
        print('outputs: ', self.outputs)
        print('params: ', self.params)

    def computePdeLoss(self):
        print('This function has to be overloaded by a child class!')
