import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def cost(Y, T):
    return -(T*np.log(Y)).sum()

def relu(X):
    return X > 0

def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    yind = np.zeros((N, K))
    for i in range(N):
        yind[i, int(Y[i])] = 1
    return yind

def softmax(X):
    expX = np.exp(X)
    return expX / expX.sum(axis=1, keepdims=True)

def forward(W1, b1, W2, b2, X):
    Z = relu(X.dot(W1) + b1)
    Y = softmax(Z.dot(W2) + b2)
    return Y, Z

DATASET_PATH = './mnist_test.csv'\

data = pd.read_csv(DATASET_PATH).to_numpy()

np.random.shuffle(data)

Y = data[:, 0]
X = data[:, 1:]

x = X[0].reshape(28, 28)

print(x.shape)

plt.imshow(x, cmap='gray')
plt.title(f'{Y[0]}')
# plt.show()

validation_split = 0.2

Xtrain = X[:-int(validation_split*len(X))]
Ytrain = Y[:-int(validation_split*len(X))]
Xtest = X[-int(validation_split*len(X)):]
Ytest = Y[-int(validation_split*len(X)):]

T = y2indicator(Ytrain)
Ytest_ind = y2indicator(Ytest)

M = 500

N, D = X.shape
K = len(set(Ytrain))

W1 = np.random.randn(D, M) / np.sqrt(D + M)
b1 = np.random.randn(M) / np.sqrt(M)
W2 = np.random.randn(M, K) / np.sqrt(M + K)
b2 = np.random.randn(K) / np.sqrt(K)

epochs = 1000
learning_rate = 0.0001

costs = []
accuracies = []

for epoch in range(epochs):
    output, hidden = forward(W1, b1, W2, b2, Xtrain)

    gradW2 = hidden.T.dot(output - T)
    gradb2 = (output - T).sum(axis=0)
    gradW1 = Xtrain.T.dot((output - T).dot(W2.T)*(hidden > 0))
    gradb1 = ((output - T).dot(W2.T)*(hidden > 0)).sum(axis=0)

    W2 -= learning_rate*gradW2
    b2 -= learning_rate*gradb2
    W1 -= learning_rate*gradW1
    b1 -= learning_rate*gradb1

    if epoch%20 == 0:
        pY_test, _ = forward(W1, b1, W2, b2, Xtest)
        # print(pY_test.shape)
        # print(Ytest_ind.shape)
        c = cost(pY_test, Ytest_ind)
        costs.append(c)
        prediction = np.argmax(pY_test, axis=1)
        acc = np.mean(prediction == Ytest)
        accuracies.append(acc)
        print(f'epoch: {epoch}, cost: {c}, accuracy: {acc}')

plt.subplot(211)
plt.plot(costs, label='costs')
plt.subplot(212)
plt.plot(accuracies, label='accuracies')
plt.legend()
plt.show()

# for epoch in range(epochs):
#     pY, Z = forward(W1, b1, W2, b2, Xtrain)

#     pY = np.array(pY, dtype=np.float64)
#     pY_Y = pY - T
#     W2 -= learning_rate*((Z.T.dot(pY_Y)))
#     b2 -= learning_rate*(((pY_Y).sum(axis=0)))
#     W1 -= learning_rate*((Xtrain.T.dot((pY_Y).dot(W2.T)*(Z > 0))))
#     b1 -= learning_rate*((((pY_Y).dot(W2.T)*(Z > 0)).sum(axis=0)))

#     if epoch%20 == 0:
#         pY_test, _ = forward(W1, b1, W2, b2, Xtest)
#         # print(pY_test.shape)
#         # print(Ytest_ind.shape)
#         c = cost(pY_test, Ytest_ind)
#         costs.append(c)
#         prediction = np.argmax(pY_test, axis=1)
#         acc = np.mean(prediction == Ytest)
#         accuracies.append(acc)
#         print(f'epoch: {epoch}, cost: {c}, accuracy: {acc}')

# plt.subplot(211)
# plt.plot(costs, label='costs')
# plt.subplot(212)
# plt.plot(accuracies, label='accuracies')
# plt.legend()
# plt.show()