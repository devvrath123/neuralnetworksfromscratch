import numpy as np

def predict(network, X, batch_size=64):
    all_predictions = []
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size].T
        output = x_batch
        for layer in network:
            output = layer.forward(output)
        batch_preds = np.argmax(output, axis=0)
        all_predictions.append(batch_preds)
    return np.concatenate(all_predictions)

def evaluate_accuracy(network, X, Y, batch_size=64):
    correct = 0
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size].T
        y_batch = Y[i:i+batch_size]
        output = x_batch
        for layer in network:
            output = layer.forward(output)
        predictions = np.argmax(output, axis=0)
        targets = np.argmax(y_batch, axis=1)
        correct += np.sum(predictions == targets)
    return (correct / len(X)) * 100

def onehot(Y, features=10):
    encoding = np.zeros((Y.size, features))
    encoding[np.arange(Y.size), Y] = 1
    return encoding

def train(network, error, error_prime, X_train, Y_train, iter = 100, lr = 0.001, batch_size = 64, details = True):
    n_samples = X_train.shape[0]
    
    for j in range(iter):
        iter_error = 0
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        for i in range(0, n_samples, batch_size):
            x_batch = X_shuffled[i:i+batch_size].T
            y_batch = Y_shuffled[i:i+batch_size].T
            output = x_batch
            for layer in network:
                output = layer.forward(output)
            iter_error += error(y_batch, output) * x_batch.shape[0]
            gradient = error_prime(y_batch, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, lr)
        iter_error /= n_samples
        if details and (j + 1) % 5 == 0:
            print(f"Iteration {j+1}/{iter}, error = {iter_error:.6f}")
    print(f"Training accuracy: {evaluate_accuracy(network, X_train, Y_train):.2f}%")