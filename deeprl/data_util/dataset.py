import numpy as np

def minibatch_shuffle(X, Y, minibatch_size=64):
    # X.shape = [num_samples, ...]
    # Y.shape = [num_samples, ...]
    num_samples = X.shape[0]

    ind = np.random.permutation(num_samples)
    X_shuffle = X[ind, :]
    Y_shuffle = Y[ind, :]

    minibatches = []

    num_full_size = int(num_samples / minibatch_size)
    for i in range(num_full_size):
        X_mini = X_shuffle[i*minibatch_size : (i+1)*minibatch_size, :]
        Y_mini = Y_shuffle[i*minibatch_size : (i+1)*minibatch_size, :]
        minibatches.append((X_mini, Y_mini))
    if num_samples % minibatch_size != 0:
        X_mini = X_shuffle[num_full_size*minibatch_size : num_samples, :]
        Y_mini = Y_shuffle[num_full_size*minibatch_size : num_samples, :]

    return minibatches

def construct_train_dev(X, Y, train_frac=.9):
    # shape = [num_samples, ...].
    num_samples = X.shape[0]
    num_train = int(num_samples * train_frac)
    ind = np.random.permutation(num_samples)
    X_train = X[ind[:num_train], :]
    Y_train = Y[ind[:num_train], :]
    X_dev = X[ind[num_train:], :]
    Y_dev = Y[ind[num_train:], :]
    return X_train, Y_train, X_dev, Y_dev
