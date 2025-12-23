import numpy as np
from dense import Dense
from activation_functions import Leaky_ReLU, Softmax
from errors import cce, cce_prime
from network import train, evaluate_accuracy
from preprocessing import preprocess

X_train, Y_train, X_test, Y_test = preprocess()

neuralnet = [
    Dense(X_train.shape[1], 128),
    Leaky_ReLU(),
    Dense(128, 64),
    Leaky_ReLU(),
    Dense(64, 10),
    Softmax()
]

print("Multi-Layer Perceptron (Dense Neural Network) with SGD")
train(neuralnet, cce, cce_prime, X_train, Y_train)

accuracy = evaluate_accuracy(neuralnet, X_test, Y_test)
print(f"Test accuracy: {accuracy:.2f}%")