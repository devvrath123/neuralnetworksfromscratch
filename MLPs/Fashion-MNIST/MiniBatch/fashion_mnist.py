import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from dense import Dense
from activation_functions import Leaky_ReLU, Softmax
from errors import cce, cce_prime
from network import train, evaluate_accuracy, predict
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

print("Multi-Layer Perceptron (Dense Neural Network) with MiniBatch + momentum based backward propagation")
train(neuralnet, cce, cce_prime, X_train, Y_train)

accuracy = evaluate_accuracy(neuralnet, X_test, Y_test)
print(f"Test Accuracy: {accuracy:.2f}%")

Y_pred = predict(neuralnet, X_test)
Y_true = np.argmax(Y_test, axis=1)

cm = confusion_matrix(Y_true, Y_pred, normalize='true')
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
cmd.plot(cmap='Purples', xticks_rotation='vertical')
plt.show()