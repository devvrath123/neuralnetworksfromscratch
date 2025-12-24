import cupy as np
from layer import Layer
from convolution import indices, im2col, col2im

class MaxPooling(Layer):
    def __init__(self, input_shape, pool_size=2, stride=2):
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.stride = stride
        depth, input_height, input_width = input_shape
        self.out_h = int((input_height - pool_size) / stride) + 1
        self.out_w = int((input_width - pool_size) / stride) + 1

    def forward(self, input):
        self.input = input
        batch_size, depth, in_h, in_w = input.shape
        reshaped_input = input[:, :, :self.out_h*self.stride, :self.out_w*self.stride].reshape(batch_size, depth, self.out_h, self.stride, self.out_w, self.stride)
        self.output = reshaped_input.max(axis=(3, 5))
        mask = (reshaped_input == self.output[:, :, :, np.newaxis, :, np.newaxis])
        self.mask = mask
        return self.output
    
    def backward(self, output_gradient, lr):
        batch_size, depth, in_h, in_w = self.input.shape
        grad_reshaped = output_gradient[:, :, :, np.newaxis, :, np.newaxis]
        input_grad_reshaped = self.mask * grad_reshaped
        input_gradient = np.zeros_like(self.input)
        input_gradient[:, :, :self.out_h*self.stride, :self.out_w*self.stride] = input_grad_reshaped.reshape(batch_size, depth, self.out_h * self.stride, self.out_w * self.stride)
        return input_gradient