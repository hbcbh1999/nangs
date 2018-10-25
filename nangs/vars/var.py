class Var:
    def __init__(self, name):
        if not isinstance(name, str): raise ValueError('variable name must be a string')
        self.name = name