from .var import Var 

class Param(Var):
    def __init__(self, name, values=[], isInput=False):
        super().__init__(name)
        self.isInput = isInput
        if not isinstance(values, list) and not isinstance(values, int) and not isinstance(values, float): 
            raise ValueError('variable values must be a list or a number')
        # check that all elements in values are numbers !
        #for value in values:
        #    if not isinstance(value, int) or not isinstance(value, float):
        #        raise ValueError('all values must be numbers')
        if isinstance(values, list): self.values = values
        else: self.values = [values]
