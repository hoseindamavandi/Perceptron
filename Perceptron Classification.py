import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv('linear_data_train.csv')

data2 = pd.read_csv('linear_data_test.csv')

X_train = np.matrix(data1.iloc[:, 0:2])
Y_train = np.array(data1.iloc[:, 2])
X_test = np.matrix(data2.iloc[:, 0:2])
Y_test = np.array(data2.iloc[:, 2])

lr = 0.001

w = np.random.rand(2, 1)
b = np.random.rand(1, 1)
print(w)

x_range = np.arange(X_train[:, 0].min(), X_train[:, 0].max(), 0.01)
y_range = np.arange(X_train[:, 1].min(), X_train[:, 1].max(), 0.01)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(-140, 60)

Error = []

for i in range(X_train.shape[0]):
    
    y_pred = np.matmul(X_train[i], w) + b
    e = Y_train[i] - y_pred

    w += lr * X_train[i, :].T * e
    b += lr * e

    Y_pred = np.matmul(X_train, w) + b
    error = np.mean(np.abs(Y_train - Y_pred))
    Error.append(error)

    ax.clear()
    x, y = np.meshgrid(x_range, y_range)
    z = w[0] * x + w[1] * y + b
    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.4)
    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='r', marker='o')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Prediction')
    plt.pause(0.1)

plt.show()
