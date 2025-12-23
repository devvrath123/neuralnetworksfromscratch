import numpy as np

def mse(y_real, y_pred):
    return np.mean(np.power((y_real - y_pred), 2))

def mse_prime(y_real, y_pred):
    return 2 * (y_pred - y_real) / np.size(y_real)

def cce(y_real, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.sum(y_real * np.log(y_pred))

def cce_prime(y_real, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1-eps)
    return y_pred - y_real