from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((output_size, 1))
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, lr):
        weights_gradient = np.dot(output_gradient, self.input.T) # dE/dW = dE/dY * X^t
        input_gradient = np.dot(self.weights.T, output_gradient) # dE/dX = W^t * dE/dY
        self.weights = self.weights - lr * weights_gradient
        self.bias = self.bias - lr * output_gradient
        return input_gradient