from layer import Layer
from activation import Activation
import numpy as np

class Leaky_ReLU(Activation):
    def __init__(self, a = 0.01):
        self.a = a
        def leaky_relu(x):
            return np.where(x >= 0, x, self.a * x)
        
        def leaky_relu_prime(x):
            return np.where(x >= 0, 1, self.a)
        
        super().__init__(leaky_relu, leaky_relu_prime)

class Softmax(Layer):
    def forward(self, input):
        E = np.exp(input - np.max(input, axis=0, keepdims=True))
        self.output = E / np.sum(E, axis=0, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, lr):
        return output_gradient
