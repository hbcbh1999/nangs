class Boco():
    def __init__(self):
        self.type = None
        self.bs = None
        self.dataset = None
        self.DataLoader = None
        
    def check(self, inputs, outputs, params):
        print('Override this function to check everything is ok')

    def summary(self):
        print('Override this function to print things')

    def setSolverParams(self, bs):
        self.bs = bs
