import cupy as np
from layer import Layer

def indices(X_shape, height, width, padding=0, stride=1):
    n, c, h, w = X_shape
    out_height = int((h + 2 * padding - height) / stride + 1)
    out_width = int((w + 2 * padding - width) / stride + 1)
    row = np.repeat(np.arange(height), width)
    row = np.tile(row, c)
    column = np.tile(np.arange(width), height)
    column = np.tile(column, c)
    row_slide = stride * np.repeat(np.arange(out_height), out_width)
    column_slide = stride * np.tile(np.arange(out_width), out_height)
    row_indices = row.reshape(-1, 1) + row_slide.reshape(1, -1)
    column_indices = column.reshape(-1, 1) + column_slide.reshape(1, -1)
    depth_indices = np.repeat(np.arange(c), height * width).reshape(1, -1)
    return (row_indices.astype(int), column_indices.astype(int), depth_indices.astype(int))

def im2col(X, height, width, padding=0, stride=1):
    X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    k, i, j = indices(X.shape, height, width, padding, stride)
    cols = X_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(height * width * X.shape[1], -1)
    return cols

def col2im(cols, X_shape, height, width, padding=0, stride=1):
    n, c, h, w = X_shape
    height_padded, width_padded = h + 2 * padding, w + 2 * padding
    X_padded = np.zeros((n, c, height_padded, width_padded), dtype=cols.dtype)
    k, i, j = indices(X_shape, height, width, padding, stride)
    cols_reshaped = cols.reshape(height * width * c, -1, c)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(X_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return X_padded
    return X_padded[:, :, padding:-padding, padding:-padding]

class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth, momentum=0.9):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.output_height = self.input_height - kernel_size + 1
        self.output_width = self.input_width - kernel_size + 1
        self.kernels = np.random.randn(depth, self.input_depth, kernel_size, kernel_size) * np.sqrt(2 / (self.input_depth * kernel_size**2))
        self.biases = np.zeros((depth, 1))
        self.momentum = momentum
        self.v_k = np.zeros_like(self.kernels)
        self.v_b = np.zeros_like(self.biases)

    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        self.kernels_col = self.kernels.reshape(self.depth, -1)
        self.input_col = im2col(self.input, self.kernel_size, self.kernel_size)
        output = np.dot(self.kernels_col, self.input_col) + self.biases
        output = output.reshape(self.depth, self.output_height, self.output_width, batch_size)
        return output.transpose(3, 0, 1, 2)
    
    def backward(self, output_gradient, lr):
        batch_size = output_gradient.shape[0]
        grad_reshape = output_gradient.transpose(1, 2, 3, 0).reshape(self.depth, -1)
        kernels_gradient = np.dot(grad_reshape, self.input_col.T).reshape(self.kernels.shape) / batch_size
        bias_gradient = np.sum(output_gradient, axis=(0, 2, 3)).reshape(self.depth, 1) / batch_size
        input_gradient_col = np.dot(self.kernels_col.T, grad_reshape)
        input_gradient = col2im(input_gradient_col, self.input.shape, self.kernel_size, self.kernel_size)
        self.v_k = self.momentum * self.v_k - lr * kernels_gradient
        self.v_b = self.momentum * self.v_b - lr * bias_gradient
        self.kernels += self.v_k
        self.biases += self.v_b
        return input_gradient