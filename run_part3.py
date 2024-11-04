import numpy as np
from nn import *

np.random.seed(0)

y = []
X = []
with open('wdbc.csv') as fh:
    csv_lines = fh.read().strip().split('\n')

    for line in csv_lines:
        line_split = [float(_) for _ in line.split(',')]

        y.append([line_split[1]])
        X.append(line_split[2:])

    y = np.array(y)
    X = np.array(X)

# splitting data
split_index = int(0.8 * len(y))
y_train = y[:split_index]
y_test = y[split_index:]
X_train = X[:split_index]
X_test = X[split_index:]

# scaling data
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std


# test
network_input_size = 30

layer_output_sizes = [10, 1]
activation_funcs = [sigmoid, sigmoid]
activation_ders = [sigmoid_der, sigmoid_der]

#layer_output_sizes = [10, 10, 1]
#activation_funcs = [sigmoid, sigmoid, sigmoid]
#activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der]

#layer_output_sizes = [10, 10, 10, 1]
#activation_funcs = [sigmoid, sigmoid, sigmoid, sigmoid]
#activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der, sigmoid_der]

#layer_output_sizes = [10, 10, 10, 10, 1]
#activation_funcs = [sigmoid, sigmoid, sigmoid, sigmoid, sigmoid]
#activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der, sigmoid_der, sigmoid_der]

layers = create_layers(network_input_size, layer_output_sizes)


epochs = 100
lr = 0.1
mo = 0

updates = [[0, 0] for _ in range(len(layer_output_sizes))]

for epoch in range(epochs):

    for i in range(len(X_train)):

        layer_grads = backpropagation(np.array([X_train[i,:]]), layers, activation_funcs, np.array([y_train[i,:]]), activation_ders)

        for k in range(len(layer_output_sizes)):
            for l in range(2):
                updates[k][l] = layers[k][l] - lr * np.mean(layer_grads[k][l], axis=0) - mo * updates[k][l]

        layers = []
        for k in range(len(layer_output_sizes)):
            layers.append((updates[k][0], updates[k][1]))

    layer_inputs, zs, predict_train = feed_forward_saver(X_train, layers, activation_funcs)
    layer_inputs, zs, predict_test = feed_forward_saver(X_test, layers, activation_funcs)

    print('Epoch ' + str(epoch+1) + '| Train accuracy: ' + str(accuracy(predict_train, y_train)) + ' | Test accuracy: ' + str(accuracy(predict_test, y_test)))

    #exit()
