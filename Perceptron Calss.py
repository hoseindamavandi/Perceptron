import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Percepton:
    def __init__(self):
        pass

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.number_class = len(np.unique(Y_train))
        x_range = np.arange(self.X_train[:, 0].min(), self.X_train[:, 0].max(), 0.01)
        y_range = np.arange(self.X_train[:, 1].min(), self.X_train[:, 1].max(), 0.01)
        x, y = np.meshgrid(x_range, y_range)

        lr = 0.01
        self.w = np.random.rand(2, 1)
        self.b = np.random.rand(1, 1)
        self.Error = []
        for i in range(self.X_train.shape[0]):
            y_pred = np.matmul(self.X_train[i], self.w) + self.b
            e = self.Y_train[i] - y_pred

            update = lr * e
            self.w += self.X_train[i, :].T * update
            self.b += update

            Y_pred = np.matmul(self.X_train, self.w) + self.b
            error = np.mean(np.abs(self.Y_train - Y_pred))
            self.Error.append(error)

            ax.clear()
            z = self.w[0] * x + self.w[1] * y + self.b
            ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.4)
            ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='r', marker='o')

            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Prediction')
            plt.pause(0.1)
        plt.show()

        return self.Error, self.w, self.b

    def predict(self, X_test):
        Y_pred = np.matmul(X_test, self.w) + self.b
        predic = np.zeros(len(Y_pred))
        for i, pred in enumerate(Y_pred):
            if pred > 0:
                predic[i] = (1)
            elif pred < 0:
                predic[i] = (-1)
        return predic

    def evaluate(self, X_test, Y_test):
        Y_pred = np.matmul(X_test, self.w) + self.b
        predic = np.zeros(len(Y_pred))
        for i, pred in enumerate(Y_pred):
            if pred > 0:
                predic[i] = 1
            elif pred < 0:
                predic[i] = (-1)
        accuracy = (predic == Y_test).sum() / len(Y_test)
        error = np.mean(np.abs(Y_test - predic))
        return error, accuracy

    def pltlost(self):
        x = np.arange(0, self.X_train.shape[0])
        plt.plot(x, self.Error, marker='o')
        plt.show()


data1 = pd.read_csv('linear_data_train.csv')
data2 = pd.read_csv('linear_data_test.csv')

X_train = np.matrix(data1.iloc[:, 0:2])
Y_train = np.array(data1.iloc[:, 2])
X_test = np.matrix(data2.iloc[:, 0:2])
Y_test = np.array(data2.iloc[:, 2])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

p = Percepton()
Error, w, b = p.fit(X_train, Y_train)
prediction = p.predict(X_train)
EvalTrain, Accuracy = p.evaluate(X_train, Y_train)
print(EvalTrain, Accuracy)
EvalTest, Accuracy = p.evaluate(X_test, Y_test)
print(EvalTest, Accuracy)
p.pltlost()