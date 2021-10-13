import numpy as np
import matplotlib.pyplot as plt

def data_generator(N):
    X = np.random.uniform(0, 4, N)
    Y =  X * 2 + np.random.normal(0, 0.5, N) + 2

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    print(X.shape)
    print(Y.shape)

    return X,Y

def sigmoid(x):
    return 1 / 1+ np.exp(-x)

N = 200
X_train , Y_train = data_generator(N)

lr = 0.001

W = np.random.rand(1, 1)
b = np.random.rand(1,1)

print(W)

fig1, (ax1,ax2) =  plt.subplots(1, 2, figsize=(12,6))

errors = []
epochs = 3

for n in range(epochs):
    for i in range(N):

        #train
        y_pred = sigmoid(np.matmul(X_train[i], W) + b)
        e = Y_train[i] - y_pred

        #Update
        W += e * lr * X_train[i]
        b += e * lr

        print(W)
        
        #Error
        Y_pred = np.matmul(X_train,W) + b
        error = np.mean(Y_train - Y_pred)
        errors.append(error)

        #Plot Data
        ax1.clear()
        ax1.set_title('Data')
        ax1.scatter(X_train , Y_train , s=1 , c='red')
        ax1.plot(X_train, Y_pred, '-c' , lw = 2)
        ax1.set_ylim(bottom=-1)
        plt.pause(0.01)

        #Plot Error
        ax2.clear()
        ax2.set_title('Loos')
        ax2.plot(errors, '-b' , lw = 1)

        plt.pause(0.01)

plt.show()
    