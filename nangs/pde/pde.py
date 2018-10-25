from ..vars import IndependentVar, DependentVar, Param

class PDE:
    def __init__(self, inputs=None, outputs=None, params=None):
        self.inputs = []
        self.outputs = []
        self.params = []
        self.bocos = []
        self.addInputs(inputs)
        self.addOutputs(outputs)
        self.addParams(params)

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
        print('outputs: ', [var.name for var in self.outputs])
        print('params: ', {var.name: (var.isInput, var.values) for var in self.params})
        print('bocos: ', [boco.type for boco in self.bocos])
        print('')

    def boco_summary(self):
        for boco in self.bocos: boco.summary() 
            
    def computePdeLoss(self):
        print('This function has to be overloaded by a child class!')
