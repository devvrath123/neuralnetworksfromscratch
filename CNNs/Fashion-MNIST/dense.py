import cupy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size, momentum=0.9):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((output_size, 1))
        self.momentum = momentum
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.bias)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, lr):
        batch_size = output_gradient.shape[1]
        weights_gradient = np.dot(output_gradient, self.input.T) / batch_size
        bias_gradient = np.sum(output_gradient, axis = 1, keepdims = True) / batch_size
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.v_w = self.momentum * self.v_w - lr * weights_gradient
        self.v_b = self.momentum * self.v_b - lr * bias_gradient
        self.weights += self.v_w
        self.bias += self.v_b
        return input_gradient