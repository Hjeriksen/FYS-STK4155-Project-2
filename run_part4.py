import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

from log_reg import *

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

lrs = [0.1, 0.01, 0.001, 0.0001]
ls = [0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
results = []
for lr in lrs:
    for l in ls:
        logistic_regressor = LogisticRegression(learning_rate=lr, epochs=30, lambda_=l)
        logistic_regressor.fit(X_train, y_train)

        y_test_predict = logistic_regressor.predict(X_test)

        a = 1 - np.sum(np.power(y_test - y_test_predict, 2)) / len(y_test)
        print(a)
        results.append(a)

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
plt.savefig('log_reg_lr_l.png', dpi=250)
plt.clf()



