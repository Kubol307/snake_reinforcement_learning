import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time 


def get_data(show=False):
    y = []
    x = []
    with open('mnist_train.csv', 'r') as f:
        for i in range(10_000):
            if i == 0:
                f.readline()
                continue 
            line = f.readline()
            y.append(int(line[0]))
            x.append([float(number)/255.0 for number in line[2:].rstrip().split(',')])

    x, y = shuffle(x, y)

    if show:
        choice = np.random.randint(0, 10_000)
        image = np.array(x[choice])
        title = f'{choice}: {y[choice]}'
        image = image.reshape((28, 28))

        plt.imshow(image)
        plt.title(title)
        plt.show()
        
    return x, y

def relu(x):
    return x*(x>0)

def softmax(x):
    expX = np.exp(x)
    return expX / expX.sum(axis=1, keepdims=True)

def softmax_0_dims(x):
    expX = np.exp(x)
    return expX / expX.sum(axis=0, keepdims=True)
  
def y2indicator(Y):
    K = len(set(Y))
    N = len(Y)
    T = np.zeros((N, K))

    for i in range(N):
        T[i, int(Y[i])] = 1
    return T

def cost(Y, T):
    return -(T*np.log(Y)).sum()

def error_rate(Y, T):
    return np.mean(T == Y)

class ANN():
    def __init__(self, M, show=False):
        X, Y = get_data(show)
        self.Xtrain = X[:8_000]
        self.Ytrain = Y[:8_000]
        self.Xtest = X[8_000:]
        self.Ytest = Y[8_000:]

        self.Xtrain = np.array(self.Xtrain)
        self.Ytrain = np.array(self.Ytrain)
        self.Xtest = np.array(self.Xtest)
        self.Ytest = np.array(self.Ytest)

        self.N, self.D = self.Xtrain.shape
        self.K = 10
        self.M = M

        self.W1 = np.random.randn(self.D, self.M) / np.sqrt(self.M + self.D)
        self.b1 = np.zeros(self.M) / np.sqrt(self.M)
        self.W2 = np.random.randn(self.M, self.K) / np.sqrt(self.M + self.K)
        self.b2 = np.zeros(self.K) / np.sqrt(self.K)

        self.W1_full = np.copy(self.W1)
        self.b1_full = np.copy(self.b1)
        self.W2_full = np.copy(self.W2)
        self.b2_full = np.copy(self.b2)

        self.W1_batch = np.copy(self.W1)
        self.b1_batch = np.copy(self.b1)
        self.W2_batch = np.copy(self.W2)
        self.b2_batch = np.copy(self.b2)        

        self.W1 = np.array(self.W1)
        self.b1 = np.array(self.b1)
        self.W2 = np.array(self.W2)
        self.b2 = np.array(self.b2)

        self.T = y2indicator(self.Ytrain)
        self.Ytest_ind = y2indicator(self.Ytest)

    def train_full_gd(self, learning_rate=1e-5, reg=0.1, epochs=10_000):
        costs = []
        best_validation_error = 0

        print(self.W1)

        start_time = time.time()

        for epoch in range(epochs):
            pY, Z = self.forward(self.Xtrain, self.W1_full, self.b1_full, self.W2_full, self.b2_full)
            # print(pY.shape)
            pY = np.array(pY, dtype=np.float64)
            pY_Y = pY - self.T
            self.W2_full -= learning_rate*(Z.T.dot(pY_Y)/self.N + reg*self.W2_full)
            self.b2_full -= learning_rate*(((pY_Y).sum(axis=0))/self.N + reg*self.b2_full)
            self.W1_full -= learning_rate*((self.Xtrain.T.dot((pY_Y).dot(self.W2_full.T)*(Z > 0)))/self.N + reg*self.W1_full)
            self.b1_full -= learning_rate*((((pY_Y).dot(self.W2_full.T)*(Z > 0)).sum(axis=0))/self.N + reg*self.b1_full)

            if epoch%20 == 0:
                pYtest, _ = self.forward(self.Xtest, self.W1_full, self.b1_full, self.W2_full, self.b2_full)
                prediction = np.argmax(pYtest, axis=1)
                prediction = np.array(prediction)
                c = cost(pYtest, self.Ytest_ind)

                if epoch == 0:
                    first_cost = c
                last_cost = c
                try:
                    print(costs[-1] - c)
                except:
                    pass
                costs.append(c)
                
                e = error_rate(self.Ytest, prediction)
                if e < best_validation_error:
                    best_validation_error = e
                print(f'error: {e}, cost: {c}, i: {epoch}')



        stop_time = time.time()
        full_time = stop_time - start_time
        cost_difference = first_cost - last_cost
        print(f'best validation error: {best_validation_error}, cost: {costs[-1]}') 
        print('Cost per second', cost_difference/full_time,)  
        plt.plot(costs, label="full")
        # plt.show()

    def train_batch_gd(self, learning_rate=1e-5, reg=0.1, epochs=10_000, batch_size=64):
        print(self.W1_batch)

        costs = []
        best_validation_error = 0
        batches = int(np.ceil(self.N / batch_size))

        start_time = time.time()

        for epoch in range(epochs):
            if epoch%20 == 0:
                pYtest, _ = self.forward(self.Xtest,self.W1_batch, self.b1_batch, self.W2_batch, self.b2_batch)
                prediction = np.argmax(pYtest, axis=1)
                prediction = np.array(prediction)
                c = cost(pYtest, self.Ytest_ind)
                
                if epoch == 0:
                    first_cost = c

                last_cost = c
                try:
                    print(costs[-1] - c)
                except:
                    pass
                costs.append(c)
                
                e = error_rate(self.Ytest, prediction)
                if e < best_validation_error:
                    best_validation_error = e
                print(f'error: {e}, cost: {c}, i: {epoch}')
            for i in range(batches):
                # pY, Z = self.forward(self.Xtrain[i*batch_size:(i+1)*batch_size])
                pY, Z = self.forward(self.Xtrain[i*batch_size:(i+1)*batch_size], self.W1_batch, self.b1_batch, self.W2_batch, self.b2_batch)
                pY = np.array(pY, dtype=np.float64)
                pY_Y = pY - self.T[i*batch_size:(i+1)*batch_size]
                self.W2_batch -= learning_rate*(Z.T.dot(pY_Y)/batch_size + reg*self.W2_batch)
                self.b2_batch -= learning_rate*(((pY_Y).sum(axis=0))/batch_size + reg*self.b2_batch)
                self.W1_batch -= learning_rate*((self.Xtrain[i*batch_size:(i+1)*batch_size].T.dot((pY_Y).dot(self.W2_batch.T)*(Z > 0)))/batch_size + reg*self.W1_batch)
                self.b1_batch -= learning_rate*((((pY_Y).dot(self.W2_batch.T)*(Z > 0)).sum(axis=0))/batch_size + reg*self.b1_batch)
                # print(i)

        stop_time = time.time()
        full_time = stop_time - start_time
        cost_difference = first_cost - last_cost
        
        print(f'best validation error: {best_validation_error}, cost: {costs[-1]}') 
        print('Cost per second', cost_difference/full_time,)  
        plt.plot(costs, label="batch")
        # plt.show()

    def forward(self, X, W1, b1, W2, b2):
        Z = relu(X.dot(W1) + b1)
        beta = Z.dot(W2) + b2
        Y = softmax(beta)
        return Y, Z
    
    def forward_0_dims(self, X):
        Z = relu(X.dot(self.W1) + self.b1)
        beta = Z.dot(self.W2) + self.b2
        Y = softmax_0_dims(beta)
        return Y, Z
    
    def predict(self, choice):

        image = np.array(self.Xtest[choice])
        pYtest, _ = self.forward_0_dims(self.Xtest[choice])
        prediction = np.argmax(pYtest)
        prediction = np.array(prediction)

        title = f'{choice}: {prediction}'
        image = image.reshape((28, 28))

        plt.imshow(image)
        plt.title(title)
        plt.show()  

np.random.seed(0)
ann = ANN(100, show=False)
ann.train_full_gd(epochs=150, learning_rate=3e-1)
ann.train_batch_gd(epochs=150, batch_size=64, learning_rate=2e-2)
plt.legend()
plt.show()
# choice = ''
# while choice.lower() != 'q':
#     choice = input("Number:")
#     ann.predict(int(choice))