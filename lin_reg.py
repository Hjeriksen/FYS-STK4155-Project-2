import autograd.numpy as np
from autograd import grad


def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def calculate_mse(a, b):
    return np.mean(np.power(a-b, 2))

def split_array(a, size):
    return np.split(a, np.arange(size,len(a),size))

def R_SGD(X_train, y_train, X_test, y_test, epochs, lr, lr_decay, mo, batch_size, l=0.0):

    np.random.seed(0)

    # initialize beta
    beta = np.random.randn(X_train.shape[1],1)
    # initialize v
    v = np.zeros((X_train.shape[1],1))

    for epoch in range(epochs):

        # shuffle before making batches
        X_train, y_train = unison_shuffled_copies(X_train,y_train)

        # split into batches
        X_train_batches = split_array(X_train, batch_size)
        y_train_batches = split_array(y_train, batch_size)

        # learning rate decay
        lr_ = lr / (1 + lr_decay * epoch)

        for X_train_batch, y_train_batch in zip(X_train_batches, y_train_batches):

            # autograd gradient
            cost = lambda beta_: np.mean(np.power(y_train_batch-X_train_batch @ beta_, 2)) + l * np.sum(np.power(beta_, 2))
            g = grad(cost)(beta)

            # analytical gradient
            #g = (2.0/X_train_batch.shape[0])*X_train_batch.T @ (X_train_batch @ beta-y_train_batch)
            #g = g + 2 * l * beta

            v = mo * v + lr_ * g.reshape(X_train_batch.shape[1], 1)
            beta -= v

    y_train_pred = X_train @ beta
    y_test_pred = X_test @ beta

    mse_train = calculate_mse(y_train_pred, y_train)
    mse_test = calculate_mse(y_test_pred, y_test)

    return mse_train, mse_test, beta

def R_ADAGRAD(X_train, y_train, X_test, y_test, epochs, lr, lr_decay, batch_size, l=0):

    np.random.seed(0)

    # initialize beta
    beta = np.random.randn(X_train.shape[1],1)
    # initialize G
    G = np.zeros((X_train.shape[1],1))
    # set epsilon
    epsilon = 10e-8

    for epoch in range(epochs):

        # shuffle before making batches
        X_train, y_train = unison_shuffled_copies(X_train,y_train)

        # split into batches
        X_train_batches = split_array(X_train, batch_size)
        y_train_batches = split_array(y_train, batch_size)

        # learning rate decay
        lr_ = lr / (1 + lr_decay * epoch)

        for X_train_batch, y_train_batch in zip(X_train_batches, y_train_batches):

            # autograd gradient
            cost = lambda beta_: np.mean(np.power(y_train_batch-X_train_batch @ beta_, 2)) + l * np.sum(np.power(beta_, 2))
            g = grad(cost)(beta)

            # analytical gradient
            #g = (2.0/X_train_batch.shape[0])*X_train_batch.T @ (X_train_batch @ beta-y_train_batch)
            #g = g + 2 * l * beta

            G = G + np.power(g, 2)
            beta -= lr * 1 / (np.sqrt(G + epsilon)) * g

    y_train_pred = X_train @ beta
    y_test_pred = X_test @ beta

    mse_train = calculate_mse(y_train_pred, y_train)
    mse_test = calculate_mse(y_test_pred, y_test)

    return mse_train, mse_test, beta

def R_RMS(X_train, y_train, X_test, y_test, epochs, lr, lr_decay, b, batch_size, l=0):

    np.random.seed(0)

    # initialize beta
    beta = np.random.randn(X_train.shape[1],1)
    # initialize s
    s = np.zeros((X_train.shape[1],1))
    # set epsilon
    epsilon = 10e-8

    for epoch in range(epochs):

        # shuffle before making batches
        X_train, y_train = unison_shuffled_copies(X_train,y_train)

        # split into batches
        X_train_batches = split_array(X_train, batch_size)
        y_train_batches = split_array(y_train, batch_size)

        # learning rate decay
        lr_ = lr / (1 + lr_decay * epoch)

        for X_train_batch, y_train_batch in zip(X_train_batches, y_train_batches):

            # autograd gradient
            cost = lambda beta_: np.mean(np.power(y_train_batch-X_train_batch @ beta_, 2)) + l * np.sum(np.power(beta_, 2))
            g = grad(cost)(beta)

            # analytical gradient
            #g = (2.0/X_train_batch.shape[0])*X_train_batch.T @ (X_train_batch @ beta-y_train_batch)
            #g = g + 2 * l * beta

            s = b * s + (1 - b) * np.power(g, 2)
            beta -= lr * g / (np.sqrt(s + epsilon))

    y_train_pred = X_train @ beta
    y_test_pred = X_test @ beta

    mse_train = calculate_mse(y_train_pred, y_train)
    mse_test = calculate_mse(y_test_pred, y_test)

    return mse_train, mse_test, beta

def R_ADAM(X_train, y_train, X_test, y_test, epochs, lr, lr_decay, b1, b2, batch_size, l=0):

    np.random.seed(0)

    # initialize beta
    beta = np.random.randn(X_train.shape[1],1)
    # initialize s
    s = np.zeros((X_train.shape[1],1))
    # initialize m
    m = np.zeros((X_train.shape[1],1))
    # set epsilon
    epsilon = 10e-8

    for epoch in range(epochs):

        # shuffle before making batches
        X_train, y_train = unison_shuffled_copies(X_train,y_train)

        # split into batches
        X_train_batches = split_array(X_train, batch_size)
        y_train_batches = split_array(y_train, batch_size)

        # learning rate decay
        lr_ = lr / (1 + lr_decay * epoch)

        t = epoch + 1

        for X_train_batch, y_train_batch in zip(X_train_batches, y_train_batches):

            # autograd gradient
            cost = lambda beta_: np.mean(np.power(y_train_batch-X_train_batch @ beta_, 2)) + l * np.sum(np.power(beta_, 2))
            g = grad(cost)(beta)

            # analytical gradient
            #g = (2.0/X_train_batch.shape[0])*X_train_batch.T @ (X_train_batch @ beta-y_train_batch)
            #g = g + 2 * l * beta

            m = b1 * m + (1 - b1) * g
            s = b2 * s + (1 - b2) * np.power(g, 2)

            m = m / (1 - np.power(b1, t))
            s = s / (1 - np.power(b2, t))

            beta -= lr * m / (np.sqrt(s) + epsilon)

    y_train_pred = X_train @ beta
    y_test_pred = X_test @ beta

    mse_train = calculate_mse(y_train_pred, y_train)
    mse_test = calculate_mse(y_test_pred, y_test)

    return mse_train, mse_test, beta

