import numpy as np
import pandas as pd
from network import onehot

def preprocess():
    train_df = (pd.read_csv('mnist_train.csv')).sample(frac=1)
    train_df = train_df.reset_index(drop=True)
    test_df = (pd.read_csv('mnist_test.csv')).sample(frac=1)
    test_df = test_df.reset_index(drop=True)
    
    X_train = np.array(train_df.drop(columns=['label'])).reshape(-1,784)
    Y_train = np.array(train_df['label']).astype(int)

    X_test = np.array(test_df.drop(columns=['label'])).reshape(-1,784)
    Y_test = np.array(test_df['label']).astype(int)

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    Y_train = onehot(Y_train)
    Y_test = onehot(Y_test)

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    return X_train, Y_train, X_test, Y_test