class Boco():
    def __init__(self):
        self.type = None

    def check(self, inputs, outputs, params):
        print('Override this function to check everything is ok')

    def summary(self):
        print('Override this function to print things')