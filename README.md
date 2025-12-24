# CNN & Multi-Layer Perceptrons from scratch
A few custom built neural networks

## A few notes

- The CNN uses a library called '**cupy**' instead of numpy. It is a drop-in replacement for numpy so that GPU compute (**dedicated** Nvidia or AMD GPUs only) can be used for numpy operations. More information can be found at [cupy.dev](cupy.dev).
- The CNN also uses the pickle library to save the model parameters after training so that it can be reused later, since training CNNs take a while, even with GPUs. There is also [already a .pkl model](https://github.com/devvrath123/neuralnetworksfromscratch/blob/main/CNNs/Fashion-MNIST/fashion-mnist-cnn.pkl) provided in the repository.
- The CNN uses an algorithm called **im2col** (image to column) instead of the convolve2d/correlate2d functions provided by scipy to vastly speed up convolution operations. More information below
- Stochastic gradient descent variants for the multi-layer perceptrons are also there, but they are slow to train.
- All neural networks use the leaky ReLU function as the activation function, the categorical cross entropy cost function and the SoftMax function for the output layer (because of multi-class classification).
- The MNIST and Fashion-MNIST datasets can be found at [Kaggle](kaggle.com/datasets). The MNIST dataset is a dataset of 60,000 28x28 grayscale images of handwritten digits in different handwriting styles, while the Fashion MNIST dataset is a dataset of 60,000 28x28 grayscale images of 10 different types of clothing items. Both datasets use a 60k to 10k train-test split

## Basic architecture

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

The variants with dropout use a dropout layer with 20% drop between the 1st and 2nd and between the 2nd and 3rd layer.

## Neural Network Performance

On the MNIST dataset (multi-layer perceptrons):
- MiniBatch Gradient Descent: 99.84% training accuracy, 97.67% test accuracy

![1](https://imgur.com/rngKV9K)

- MiniBatch Gradient Descent + Dropout: 99.97% training accuracy, 98.22% test accuracy
- MiniBatch Gradient Descent + Dropout + L2 regularisation: 99.99% training accuracy, 98.21% test accuracy

On the Fashion MNIST dataset:
- MiniBatch Gradient Descent: 94.7% training accuracy, 89.21% test accuracy
- MiniBatch Gradient Descent + Dropout: 95.4% training accuracy, 90.07% test accuracy
- MiniBatch Gradient Descent + Dropout + L2 regularisation: 95.6% training accuracy, 90.18% test accuracy
- Convolutional Neural Network: 97.92% training accuracy, 94.17% test accuracy

As visible, multi-layer perceptrons perform very well on the MNIST dataset, but struggle with the Fashion-MNIST dataset. This is because the images have more complex features, and sometimes the same shape (ex: Shirt and Pullover), making it difficult for MLPs to differentiate between them. The network ends up overfitting instead. Convolutional layers are needed to differentiate complex images better, and as shown above, the CNN far outperforms the MLPs. A more complex and scaled up CNN architecture can be used to get even better results
