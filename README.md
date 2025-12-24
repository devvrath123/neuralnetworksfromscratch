# CNN & Multi-Layer Perceptrons from scratch
A few custom built neural networks

## A few notes

- The CNN uses a library called '**cupy**' instead of numpy. It is a drop-in replacement for numpy so that GPU compute (**dedicated** Nvidia or AMD GPUs only) can be used for numpy operations. More information can be found at [cupy.dev](cupy.dev).
- The CNN also uses the pickle library to save the model parameters after training so that it can be reused later, since training CNNs take a while, even with GPUs. There is also [already a .pkl model](https://github.com/devvrath123/neuralnetworksfromscratch/blob/main/CNNs/Fashion-MNIST/fashion-mnist-cnn.pkl) provided in the repository.
- The CNN uses an algorithm called **im2col** (image to column) instead of the convolve2d/correlate2d functions provided by scipy to vastly speed up convolution operations. More information in later pages
- Stochastic gradient descent variants for the multi-layer perceptrons are also there, but they are slow to train.
- All neural networks use the leaky ReLU function as the activation function, the categorical cross entropy cost function and the SoftMax function for the output layer (because of multi-class classification)
- The MNIST and Fashion-MNIST datasets can be found at [Kaggle](kaggle.com/datasets).

## Basic structure

The convolutional neural network uses the following structure:

1. 2 conv layers (3x3 kernels, 32 and 64 filters respectively)
2. Max pooling layer
3. Dropout layer (25% drop)
4. 1 conv layer (3x3 kernel, 128 filters)
5. Max pooling layer
6. Dropout layer (25% drop)
7. Flatten/Reshape layer
8. 1 dense layer (3200 neurons to 512 neurons)
9. Dropout layer (50% drop)
10. 1 dense layer (512 neurons to 10 neurons)
11. SoftMax layer

The multi-layer perceptrons use the following basic structure:

1. 1st dense layer (784 neurons to 128 neurons)
2. 2nd dense layer (128 neurons to 64 neurons)
3. 3rd dense layer (64 neurons to 10 neurons)
4. SoftMax layer

The dropout variants use a dropout layer with 20% drop between the 1st and 2nd and between the 2nd and 3rd layer
