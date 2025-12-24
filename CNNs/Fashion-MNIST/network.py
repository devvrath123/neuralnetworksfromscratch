import cupy as np
import pickle

def predict(network, X, batch_size=64):
    if X.ndim == 3:
        X = X.reshape(1, *X.shape)
    all_predictions = []
    for layer in network:
        layer.training = False   
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        output = x_batch
        for layer in network:
            output = layer.forward(output)
        batch_preds = np.argmax(output, axis=0)
        all_predictions.append(batch_preds)
    return np.concatenate(all_predictions)

def evaluate_accuracy(network, X, Y, batch_size=64):
    correct = 0
    for layer in network:
        layer.training = False
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]
        output = x_batch
        for layer in network:
            output = layer.forward(output)
        predictions = np.argmax(output, axis=0)
        targets = np.argmax(y_batch, axis=1)
        correct += np.sum(predictions == targets)
    return (correct / len(X)) * 100

def train(network, error, error_prime, X_train, Y_train, iter=100, lr=0.01, batch_size=64, decay=0.01, details=True):
    n_samples = X_train.shape[0]
    initial_lr = lr
    for j in range(iter):
        current_lr = initial_lr / (1 + decay * j)
        iter_loss = 0
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        for layer in network:
            layer.training = True
        for i in range(0, n_samples, batch_size):
            x_batch = X_shuffled[i:i+batch_size]
            y_batch = Y_shuffled[i:i+batch_size].T
            output = x_batch
            for layer in network:
                output = layer.forward(output)
            batch_loss = error(y_batch, output)
            iter_loss += batch_loss * x_batch.shape[0]
            gradient = error_prime(y_batch, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, current_lr)
        avg_loss = iter_loss / n_samples
        if details == True and (j + 1) % 5 == 0:
            print(f"Iteration: {j+1}/{iter}, error: {avg_loss:.4f}")
    print(f"Final training accuracy: {evaluate_accuracy(network, X_train, Y_train):.2f}%")

def save_model(network, filename):
    with open(filename, 'wb') as f:
        pickle.dump(network, f)
    print(f"Model saved as {filename}")

def load_model(filename):
    with open(filename, 'rb') as f:
        network = pickle.load(f)
    print(f"Model loaded from {filename}")
    return network