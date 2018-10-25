from .var import Var

class DependentVar(Var):
    def __init__(self, name):
        super().__init__(name)