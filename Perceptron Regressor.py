import numpy as np
import matplotlib.pyplot as plt

def data_generator(N):
    X = np.random.uniform(0, 40, N)
    Y =  X * 2 + np.random.normal(0, 2, N)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    print(X.shape)
    print(Y.shape)

    return X,Y


N = 200
X_train , Y_train = data_generator(N)

#plt.scatter(X_train, Y_train)
#plt.show()

lr = 0.0001

W = np.random.rand(1, 1)

print(W)

fig, ax =  plt.subplots()

for i in range(N):

    #train
    y_pred = np.matmul(X_train[i], W)
    e = Y_train[i] - y_pred
    W = W + e * lr * X_train[i]
    print(W)
    
    #plot
    Y_pred = np.matmul(X_train, W)
    ax.clear()
    plt.scatter(X_train , Y_train , c='red')
    ax.plot(X_train, Y_pred, c = 'blue' , lw = 2)
    plt.pause(0.01)
    