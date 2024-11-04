from nn import *


np.random.seed(0)

# data
n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x+2*x**2+np.random.randn(n,1)

X = np.c_[np.ones((n,1)), x, x*x]

# splitdataset
split_index = int(0.8 * len(y))
z_train = y[:split_index]
z_test = y[split_index:]
X_train = X[:split_index]
X_test = X[split_index:]

# test
network_input_size = 3
layer_output_sizes = [5, 1]

activation_funcs = [LReLU, LReLU]
activation_ders = [LReLU_der, LReLU_der]

layers = create_layers(network_input_size, layer_output_sizes)


epochs = 3000
lr = 0.01
mo = 0

a, b, c, d = 0, 0, 0, 0

for epoch in range(epochs):

    for i in range(len(X_train)):

        layer_grads = backpropagation(np.array([X_train[i,:]]), layers, activation_funcs, np.array([z_train[i,:]]), activation_ders)

        #print(layer_grads)
        #exit()

        '''
        print(layers[0][0].shape)
        print(layers[0][1].shape)
        print(layers[1][0].shape)
        print(layers[1][1].shape)
        print('grads')
        print(layer_grads[0][0].shape)
        print(layer_grads[0][1].shape)
        print(layer_grads[1][0].shape)
        print(layer_grads[1][1].shape)
        '''

        a = layers[0][0] - lr * np.mean(layer_grads[0][0], axis=0) - mo * a
        b = layers[0][1] - lr * np.mean(layer_grads[0][1], axis=0) - mo * b
        c = layers[1][0] - lr * np.mean(layer_grads[1][0], axis=0) - mo * c
        d = layers[1][1] - lr * np.mean(layer_grads[1][1], axis=0) - mo * d

        layers = [
            (a, b),
            (c, d)
        ]

    layer_inputs, zs, predict_train = feed_forward_saver(X_train, layers, activation_funcs)
    layer_inputs, zs, predict_test = feed_forward_saver(X_test, layers, activation_funcs)
    print('Epoch ' + str(epoch+1) + '| Train MSE: ' + str(mse(predict_train, z_train)) + ' | Test MSE: ' + str(mse(predict_test, z_test)))
