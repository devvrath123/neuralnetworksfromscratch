import cupy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.batch_size = input.shape[0]
        return input.reshape(self.batch_size, -1).T

    def backward(self, output_gradient, lr):
        return output_gradient.T.reshape(self.batch_size, *self.input_shape)