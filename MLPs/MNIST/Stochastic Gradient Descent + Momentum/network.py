import numpy as np

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def evaluate_accuracy(network, X, Y):
    predictions_raw = predict(network, X.T)
    predictions_labels = np.argmax(predictions_raw, axis = 0)
    Y_reallabels = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions_labels == Y_reallabels) * 100
    return accuracy

def onehot(Y, features=10):
    encoding = np.zeros((Y.size, features))
    encoding[np.arange(Y.size), Y] = 1
    return encoding

def train(network, error, error_prime, X_train, Y_train, iter = 100, lr = 0.001, details = True):
    for i in range(iter):
        diff = 0
        for x,y in zip(X_train, Y_train):
            x = x.reshape(-1, 1) 
            y = y.reshape(-1, 1)
            output = predict(network, x)
            diff = diff + error(y, output)
            gradient = error_prime(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, lr)
        diff = diff / len(X_train)
        if details:
            print(f"{i + 1}/{iter}, error = {diff}")
    print(f"Training accuracy: {evaluate_accuracy(network, X_train, Y_train):.2f}%")
