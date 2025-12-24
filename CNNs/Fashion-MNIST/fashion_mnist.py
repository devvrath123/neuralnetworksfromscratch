import cupy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from dense import Dense
from convolution import Convolution
from dropout import Dropout
from pooling import MaxPooling
from reshape import Reshape
from activation_functions import Leaky_ReLU, Softmax
from errors import cce, cce_prime
from network import train, evaluate_accuracy, predict, load_model, save_model
from preprocessing import preprocess

X_train, Y_train, X_test, Y_test = preprocess()

neuralnet = [
    Convolution((1, 28, 28), 3, 32),
    Leaky_ReLU(),
    Convolution((32, 26, 26), 3, 64),
    Leaky_ReLU(),
    MaxPooling((64, 24, 24)),
    Dropout(0.25),
    
    Convolution((64, 12, 12), 3, 128),
    Leaky_ReLU(),
    MaxPooling((128, 10, 10)),
    Dropout(0.25),
    
    Reshape((128, 5, 5), (3200, 1)),
    Dense(3200, 512),
    Leaky_ReLU(),
    Dropout(0.5),
    Dense(512, 10),
    Softmax()
]

print("Fast CNN with MiniBatch GD + max pooling + dropout + lr decay + momentum based backward propagation")
# Uncomment lines 40 and 41 and comment lines 44 and 46 to train the model

#train(neuralnet, cce, cce_prime, X_train, Y_train, iter=40, lr=0.005, decay=0.001)
#save_model(neuralnet, "fashion-mnist-cnn.pkl")
saved_model = load_model("fashion-mnist-cnn.pkl")

train_accuracy = evaluate_accuracy(saved_model, X_train, Y_train)
test_accuracy = evaluate_accuracy(saved_model, X_test, Y_test)
print(f"Training accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

Y_pred_classes = predict(saved_model, X_test)
Y_true_classes = np.argmax(Y_test, axis=1)
y_true_cpu = Y_true_classes.get()
y_pred_cpu = Y_pred_classes.get()

cm = confusion_matrix(y_true_cpu, y_pred_cpu, normalize='true')
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
cmd.plot(cmap='Purples', xticks_rotation='vertical')
plt.show()