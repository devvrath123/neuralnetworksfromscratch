# CNN & Multi-Layer Perceptrons from scratch
A few custom built neural networks

## A few notes

- The CNN uses a library called '**cupy**' instead of numpy. It is a drop-in replacement for numpy so that GPU compute (**dedicated** Nvidia or AMD GPUs only) can be used for numpy operations. More information can be found at [cupy.dev](cupy.dev).
- The CNN also uses the pickle library to save the model parameters after training so that it can be reused later, since training CNNs take a while, even with GPUs. There is also [already a .pkl model](https://github.com/devvrath123/neuralnetworksfromscratch/blob/main/CNNs/Fashion-MNIST/fashion-mnist-cnn.pkl) provided in the repository.
- If you want to try out the convolutional neural network on your own, you can check out this [Kaggle notebook](https://www.kaggle.com/code/devvrath123/fashion-mnist-cnn). Use a GPU as the accelerator in Kaggle before you run the notebook, otherwise the code will not execute correctly.
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

## Working explanation

The multi-layer perceptrons use fully connected dense layers, ie every neuron in a layer is connected to every neuron in the previous/next layers. The connection between each neuron is represented by a 'weight', and each neuron is biased. The output layer is a SoftMax layer because this is a multinomial classification problem and the final probabilities need to sum up to 1. The output of a layer is the input of the next layer.

Dropout layers turn off a random percentage of the neuron connections between 2 layers, only during training. This helps the network generalise, not memorise.

On the forward pass, the following input is sent through all the layers and into the activation functions:

$$Y = W \cdot X + b$$

Where W is the matrix of weights and b is the matrix of biases. X is the flattened input image vector. The activation function, the Leaky ReLU function is given below:

$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases}
$$

For the backward pass, we calculate the error at the output using the categorical cross entropy cost function and perform gradient descent. Gradients are also calculated for the input, the weights and the biases and updated accordingly. Since we are using a SoftMax output layer coupled with the CCE cost function, the output gradient simplifies to:

$$\frac{\partial E}{\partial y_i} = \hat{y}_i - y_i$$

Weights gradient:

$$\frac{\partial E}{\partial W} = \frac{\partial E}{\partial Y} \cdot X$$

Bias gradient:

$$\frac{\partial E}{\partial b} = \frac{\partial E}{\partial Y}$$

Input gradient:

$$\frac{\partial E}{\partial X} = W^T \cdot \frac{\partial E}{\partial Y}$$

Since we are using momentum based backward propagation, there is a 'velocity' term, which determines how quickly the network converges based on a 'momentum' factor. The velocity acts like an accumulated gradient, helping the network converge faster toward a 'deeper' minima, and avoiding oscillations by skipping over local minima. The gradient descent will look like this:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$$

$$\theta_{t+1} = \theta_t - v_t$$

Where $v_t$ is the velocity (or accumulated gradient) at time step $t$. $\gamma$ is the momentum coefficient, determining how much previous velocity is retained. $\eta$ is the learning rate. $$\nabla_\theta J(\theta)$$ is the gradient of the cost function w.r.t $$\theta$$, which is either the weights or biases

Since we are also using MiniBatch gradient descent, the weight and bias gradients are divided by the batch size. MiniBatch gradient descent calculates the gradients over a random batch of the data instead of iterating through every single sample in the training data. The weights and biases are updated once per batch instead of sample, making training much faster and computationally efficient.

In the case of convolutional neural networks, we have the same $Y = W \cdot X + b$ forward pass, but instead of a weights matrix it is a matrix of kernels/filters, depending on how many filters are used, so essentially a matrix of matrices. The connection between each neuron in the convolutional layers is represented instead by a convolution with a kernel, rather than a dot product with a weight (as seen with MLPs). $X$ is the matrix of input images with the shape (batch size, channel, height, width). The channel is just 1 in our case because the data is made up of grayscale images. The output shape changes after every convolution layer. Height of the result is given by: $H_{out} = H_{in} - K + 1$, where K is the kernel size. The same formula applies for the width, where we simply substitute width in place of height.

Instead of using nested for loops in combination with the convolve2d/correlate2d functions from the scipy library to convolve the filters and the input images, we use the im2col algorithm. This is much faster because it converts the convolution operation into a single matrix multiplication. It works by extracting the positions of the input image matrix that the filter would slide over and makes them columns of a matrix. That matrix is then multiplied with the flattened kernel. The result is a convolution. While it is computationally faster, it does use more memory than the alternative.

Some max pooling layers are also used in our CNN architecture. They essentially perform dimensionality reduction on the input. They slide a 2x2 mask with a stride of 2 over the input and take the maximum value. This basically preserves the most significant/important features in the image and discards the rest, helping the network generalise better. Since we are using a 2x2 mask with stride = 2, it halves the size, improving speed.

The output of the convolutional layers must be reshaped before it goes into the dense layers, hence there is a reshape class implemented.

For the backward pass:

Weights gradient:

$$\frac{\partial E}{\partial W} = X * \frac{\partial L}{\partial Y}$$

Bias gradient:

$$\frac{\partial E}{\partial b} = \frac{\partial E}{\partial Y}$$

Input gradient:

$$\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} * W_{rot180}$$

For the equations of the momentum gradient descent, the same generalisation given above applies here as well. For the backward propagation, we use the col2im algorithm, which is the im2col algorithm reversed. It takes the gradients calculated for the flattened columns and puts them together back into the shape of the original image. In the forward pass there is an overlap between the pixel values when we convolve with the kernel (because stride = 1), therefore in the backward pass the col2im algorithm takes the gradients corresponding to those overlapping pixels and sums them to find the total gradient for that pixel. Where there is no overlap, its simply the individual gradient for that pixel. A matrix of the original image shape is reconstructed this way, with each value corresponding to the gradient for every pixel. The backward pass through the dense layers works as with MLPs.

## Neural Network Performance

On the MNIST dataset (multi-layer perceptrons):
- MiniBatch Gradient Descent: 99.84% training accuracy, 97.67% test accuracy
- MiniBatch Gradient Descent + Dropout: 99.97% training accuracy, 98.22% test accuracy
- MiniBatch Gradient Descent + Dropout + L2 regularisation: 99.99% training accuracy, 98.21% test accuracy

On the Fashion MNIST dataset:
- MiniBatch Gradient Descent: 94.7% training accuracy, 89.21% test accuracy
- MiniBatch Gradient Descent + Dropout: 95.4% training accuracy, 90.07% test accuracy
- MiniBatch Gradient Descent + Dropout + L2 regularisation: 95.6% training accuracy, 90.18% test accuracy
- Convolutional Neural Network: 97.92% training accuracy, 94.17% test accuracy

As visible, multi-layer perceptrons perform very well on the MNIST dataset, but struggle with the Fashion-MNIST dataset. There is a slight improvement with the addition of dropout and L2 regularisation, but performance is still poor. This is because the images have more complex features, and sometimes the items have the same shape (ex: Shirt and Pullover), making it difficult for MLPs to differentiate between them. The network ends up overfitting instead. Convolutional layers are needed to differentiate complex images better, and as shown above, the CNN far outperforms the MLPs, with a much smaller overfitting gap. Performance is greatly improved by Dropout and Max Pooling layers. A more complex and scaled up CNN architecture can be used to get even better results.

## Confusion Matrices

For MNIST:
- MiniBatch Gradient Descent
  
  ![1](https://i.imgur.com/Mp7Y1j0.png)
  
- MiniBatch Gradient Descent + Dropout
  
  ![2](https://i.imgur.com/M5kuEdO.png)
  
- MiniBatch Gradient Descent + Dropout + L2 regularisation
  
  ![3](https://i.imgur.com/A0y6FJg.png)

For Fashion-MNIST:
- MiniBatch Gradient Descent

  ![1](https://i.imgur.com/eGy05IZ.png)
  
- MiniBatch Gradient Descent + Dropout

  ![2](https://i.imgur.com/bUONcoX.png)
  
- MiniBatch Gradient Descent + Dropout + L2 regularisation

  ![3](https://i.imgur.com/A05orZG.png)
  
- Convolutional Neural Network

  ![4](https://i.imgur.com/Chmxz3m.png)
