class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.training = True
    def forward(self, input):
        pass
    def backward(self, output_gradient, lr):
        pass