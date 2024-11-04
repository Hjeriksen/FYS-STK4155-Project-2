import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

from lin_reg import *


np.random.seed(2000)

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

print('OLS analytical beta')
print(np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ z_train))
print('Ridge analytical beta')
print(np.linalg.pinv(X_train.T @ X_train + 0.1 * np.ones(3)) @ (X_train.T @ z_train))


lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
results = []
for _ in lrs:
    r = R_SGD(
        X_train, z_train,X_test, z_test,
        epochs=3000, lr=_, lr_decay=0,
        mo=0, batch_size=X_train.shape[0],
        l=0.0
    )
    results.append(r[1])

#plt.title('GD OLS test MSE in relation to learning rate')
plt.scatter(lrs, results)
plt.xlabel('LR')
plt.ylabel('Test MSE')
plt.xscale('log')
plt.savefig('ols_gd_lr.png', dpi=250)
plt.clf()

mos = [0.1, 0.01, 0.001, 0.0001, 0.00001]
results = []
for _ in lrs:
    r = R_SGD(
        X_train, z_train,X_test, z_test,
        epochs=3000, lr=0.001, lr_decay=0,
        mo=_, batch_size=X_train.shape[0],
        l=0.0
    )
    results.append(r[1])

#plt.title('GD OLS test MSE in relation to momentum')
plt.scatter(lrs, results)
plt.xlabel('Momentum')
plt.ylabel('Test MSE')
plt.xscale('log')
plt.savefig('ols_gd_momentum.png', dpi=250)
plt.clf()

epochs = [10, 100, 1000, 10000]
results = []
for _ in epochs:
    r = R_SGD(
        X_train, z_train,X_test, z_test,
        epochs=_, lr=0.01, lr_decay=0,
        mo=0, batch_size=X_train.shape[0],
        l=0.0
    )
    results.append(r[1])

#plt.title('GD OLS test MSE in relation to the number of epochs')
plt.scatter(epochs, results)
plt.xlabel('Number of epochs')
plt.ylabel('Test MSE')
plt.xscale('log')
plt.savefig('ols_gd_epochs.png', dpi=250)
plt.clf()

lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
results = []
for _ in lrs:
    r = R_SGD(
        X_train, z_train,X_test, z_test,
        epochs=1000, lr=_, lr_decay=0,
        mo=0, batch_size=X_train.shape[0],
        l=0.1
    )
    results.append(r[1])

#plt.title('GD Ridge test MSE in relation to learning rate')
plt.scatter(lrs, results)
plt.xlabel('LR')
plt.ylabel('Test MSE')
plt.xscale('log')
plt.savefig('ridge_gd_lr.png', dpi=250)
plt.clf()

ls = [0.1, 0.01, 0.001, 0.0001, 0.00001]
results = []
for _ in ls:
    r = R_SGD(
        X_train, z_train,X_test, z_test,
        epochs=1000, lr=0.001, lr_decay=0,
        mo=0, batch_size=X_train.shape[0],
        l=_
    )
    results.append(r[1])

#plt.title('GD Ridge test MSE in relation to lambda')
plt.scatter(lrs, results)
plt.xlabel(r'$\lambda$')
plt.ylabel('Test MSE')
plt.xscale('log')
plt.savefig('ridge_gd_lambda.png', dpi=250)
plt.clf()

lr_decays = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
results = []
for _ in lr_decays:
    r = R_SGD(
        X_train, z_train,X_test, z_test,
        epochs=1000, lr=0.001, lr_decay=_,
        mo=0, batch_size=X_train.shape[0],
        l=0.01
    )
    results.append(r[1])

#plt.title('GD Ridge test MSE in relation to learnig rate decay')
plt.scatter(lr_decays, results)
plt.xlabel(r'LR decay')
plt.ylabel('Test MSE')
plt.xscale('log')
plt.savefig('ridge_gd_lrdecay.png', dpi=250)
plt.clf()

batch_sizes = [1, 16, 32, 64, 128, 256, X_train.shape[0]]
results = []
for _ in batch_sizes:
    r = R_SGD(
        X_train, z_train,X_test, z_test,
        epochs=1000, lr=0.001, lr_decay=0,
        mo=0, batch_size=_,
        l=0.01
    )
    results.append(r[1])

#plt.title('GD Ridge test MSE in relation to batch size')
plt.scatter(batch_sizes, results)
plt.xlabel(r'Batch size')
plt.ylabel('Test MSE')
plt.xscale('log')
plt.savefig('ridge_gd_batchsize.png', dpi=250)
plt.clf()

lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
results = []
for _ in lrs:
    r1 = R_SGD(
        X_train, z_train, X_test, z_test,
        epochs=1000, lr=_, lr_decay=0,
        mo=0, batch_size=64,
        l=0.01
    )
    r2 = R_ADAGRAD(
        X_train, z_train, X_test, z_test,
        epochs=1000, lr=_, lr_decay=0,
        batch_size=64,
        l=0.01
    )
    r3 = R_RMS(
        X_train, z_train, X_test, z_test,
        epochs=1000, lr=_, lr_decay=0,
        b=0.9, batch_size=64,
        l=0.01
    )
    r4 = R_ADAM(
        X_train, z_train, X_test, z_test,
        epochs=1000, lr=_, lr_decay=0,
        b1=0.2, b2=0.2, batch_size=64,
        l=0.01
    )

    results.append([r1[1], r2[1], r3[1], r4[1]])


#plt.title('Ridge test MSE with different optimizers in relation to learnig rate')
plt.scatter(lrs, [_[0] for _ in results], label='SGD')
plt.scatter(lrs, [_[1] for _ in results], label='AdaGrad')
plt.scatter(lrs, [_[2] for _ in results], label='RMS Prop')
plt.scatter(lrs, [_[3] for _ in results], label='Adam')
plt.legend(loc='best')
plt.xlabel(r'LR')
plt.ylabel('Test MSE')
plt.xscale('log')
plt.savefig('ridge_gd_comp.png', dpi=250)
plt.clf()

lrs = [0.1, 0.01, 0.001, 0.0001]
ls = [0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
results = []
for lr in lrs:
    for l in ls:
        r = R_SGD(
            X_train, z_train,X_test, z_test,
            epochs=1000, lr=lr, lr_decay=0,
            mo=0, batch_size=64,
            l=l
        )
        results.append(r[1])

print(results)
data = np.array(results).reshape(len(lrs),len(ls))

fig, ax = plt.subplots()
im = ax.imshow(data)

# Show all ticks and label them with the respective list entries
ax.set_yticks(np.arange(len(lrs)), labels=lrs)
ax.set_xticks(np.arange(len(ls)), labels=ls)

ax.set_ylabel('LR')
ax.set_xlabel(r'$\lambda$')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(lrs)):
    for j in range(len(ls)):
        text = ax.text(j, i, '{:.2f}'.format(data[i, j]),
                    ha="center", va="center", color="w")

fig.tight_layout()
#plt.title('SGD Ridge test MSE with respect to the learning rate and lambda')
plt.savefig('ridge_gd_lr_l.png', dpi=250)
plt.clf()
