import cupy as np
from layer import Layer

class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, input):
        if self.training:
            self.mask = (np.random.rand(*input.shape) > self.rate) / (1 - self.rate)
            return input * self.mask
        return input

    def backward(self, output_gradient, lr):
        if self.mask is None:
            return output_gradient
        return output_gradient * self.mask