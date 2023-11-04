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

        self.T = y2indicator(self.Ytrain)
        self.Ytest_ind = y2indicator(self.Ytest)

    def train_full_gd(self, learning_rate=1e-5, reg=0.1, epochs=10_000):
        W1 = self.W1.copy()
        b1 = self.b1.copy()
        W2 = self.W2.copy()
        b2 = self.b2.copy()
        print('full gd')
        print(W1)

        costs = []
        best_validation_error = 0

        start_time = time.time()

        for epoch in range(epochs):
            if epoch%20 == 0:
                pYtest, _ = self.forward(self.Xtest, W1, b1, W2, b2)
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
            pY, Z = self.forward(self.Xtrain, W1, b1, W2, b2)
            # print(pY.shape)
            pY = np.array(pY, dtype=np.float64)
            pY_Y = pY - self.T
            W2 -= learning_rate*(Z.T.dot(pY_Y)/self.N + reg*W2)
            b2 -= learning_rate*(((pY_Y).sum(axis=0))/self.N + reg*b2)
            W1 -= learning_rate*((self.Xtrain.T.dot((pY_Y).dot(W2.T)*(Z > 0)))/self.N + reg*W1)
            b1 -= learning_rate*((((pY_Y).dot(W2.T)*(Z > 0)).sum(axis=0))/self.N + reg*b1)

        stop_time = time.time()
        full_time = stop_time - start_time
        cost_difference = first_cost - last_cost
        print(f'best validation error: {best_validation_error}, cost: {costs[-1]}') 
        print('Cost per second', cost_difference/full_time,)  
        plt.plot(costs, label="full")
        # plt.show()

    def train_batch_gd(self, learning_rate=1e-5, reg=0.1, epochs=10_000, batch_size=64):
        W1 = self.W1.copy()
        b1 = self.b1.copy()
        W2 = self.W2.copy()
        b2 = self.b2.copy()
        print(W1)

        costs = []
        best_validation_error = 0
        batches = int(np.ceil(self.N / batch_size))

        start_time = time.time()

        for epoch in range(epochs):
            if epoch%20 == 0:
                pYtest, _ = self.forward(self.Xtest,W1, b1, W2, b2)
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
                pY, Z = self.forward(self.Xtrain[i*batch_size:(i+1)*batch_size], W1, b1, W2, b2)
                pY = np.array(pY, dtype=np.float64)
                pY_Y = pY - self.T[i*batch_size:(i+1)*batch_size]
                W2 -= learning_rate*(Z.T.dot(pY_Y)/batch_size + reg*W2)
                b2 -= learning_rate*(((pY_Y).sum(axis=0))/batch_size + reg*b2)
                W1 -= learning_rate*((self.Xtrain[i*batch_size:(i+1)*batch_size].T.dot((pY_Y).dot(W2.T)*(Z > 0)))/batch_size + reg*W1)
                b1 -= learning_rate*((((pY_Y).dot(W2.T)*(Z > 0)).sum(axis=0))/batch_size + reg*b1)
                # print(i)

        stop_time = time.time()
        full_time = stop_time - start_time
        cost_difference = first_cost - last_cost
        
        print(f'best validation error: {best_validation_error}, cost: {costs[-1]}') 
        print('Cost per second', cost_difference/full_time,)  
        plt.plot(costs, label="batch")
        # plt.show()

    def train_batch_gd_nesterov_momentum_RMS(self, learning_rate=1e-5, reg=0.1, epochs=10_000, batch_size=64, label='nesterov rms', decay_rate=0.999):
        W1 = self.W1.copy()
        b1 = self.b1.copy()
        W2 = self.W2.copy()
        b2 = self.b2.copy()
        print('nesterov rms')
        print(W1)

        costs = []
        best_validation_error = 0
        batches = int(np.ceil(self.N / batch_size))

        start_time = time.time()

        vW2 = 0
        vb2 = 0
        vW1 = 0
        vb1 = 0
        mu = 0.9

        cacheW1 = 1
        cacheb1 = 1
        cacheW2 = 1
        cacheb2 = 1
        epsilon = 0.000001

        for epoch in range(epochs):
            if epoch%20 == 0:
                pYtest, _ = self.forward(self.Xtest,W1, b1, W2, b2)
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
                pY, Z = self.forward(self.Xtrain[i*batch_size:(i+1)*batch_size], W1, b1, W2, b2)
                pY = np.array(pY, dtype=np.float64)
                pY_Y = pY - self.T[i*batch_size:(i+1)*batch_size]

                gW2 = (Z.T.dot(pY_Y)/batch_size + reg*W2)
                gb2 = (((pY_Y).sum(axis=0))/batch_size + reg*b2)
                gW1 = ((self.Xtrain[i*batch_size:(i+1)*batch_size].T.dot((pY_Y).dot(W2.T)*(Z > 0)))/batch_size + reg*W1)
                gb1 = ((((pY_Y).dot(W2.T)*(Z > 0)).sum(axis=0))/batch_size + reg*b1)

                cacheW2 = decay_rate*cacheW2 + (1 - decay_rate)*gW2*gW2
                cacheb2 = decay_rate*cacheb2 + (1 - decay_rate)*gb2*gb2
                cacheW1 = decay_rate*cacheW1 + (1 - decay_rate)*gW1*gW1
                cacheb1 = decay_rate*cacheb1 + (1 - decay_rate)*gb1*gb1

                vW2 = mu*vW2 + (1 - mu)*learning_rate*gW2/(np.sqrt(cacheW2) + epsilon)
                vb2 = mu*vb2 + (1 - mu)*learning_rate*gb2/(np.sqrt(cacheb2) + epsilon)
                vW1 = mu*vW1 + (1 - mu)*learning_rate*gW1/(np.sqrt(cacheW1) + epsilon)
                vb1 = mu*vb1 + (1 - mu)*learning_rate*gb1/(np.sqrt(cacheb1) + epsilon)

                W2 -= vW2
                b2 -= vb2
                W1 -= vW1
                b1 -= vb1
                # print(i)

        stop_time = time.time()
        full_time = stop_time - start_time
        cost_difference = first_cost - last_cost
        
        print(f'best validation error: {best_validation_error}, cost: {costs[-1]}') 
        print('Cost per second', cost_difference/full_time,)  
        plt.plot(costs, label=label)
        # plt.show()

    def train_batch_gd_Adam(self, learning_rate=0.0001, reg=0.1, epochs=10_000, batch_size=64, label='Adam'):
        W1 = self.W1.copy()
        b1 = self.b1.copy()
        W2 = self.W2.copy()
        b2 = self.b2.copy()
        print('nesterov rms')
        print(W1)

        costs = []
        best_validation_error = 0
        batches = int(np.ceil(self.N / batch_size))

        start_time = time.time()

        B1 = 0.99
        B2 = 0.999

        epsilon = 1e-8

        vW2 = 0
        vb2 = 0
        vW1 = 0
        vb1 = 0

        mW2 = 0
        mb2 = 0
        mW1 = 0
        mb1 = 0

        t = 1

        for epoch in range(epochs):
            if epoch%20 == 0:
                pYtest, _ = self.forward(self.Xtest,W1, b1, W2, b2)
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
                pY, Z = self.forward(self.Xtrain[i*batch_size:(i+1)*batch_size], W1, b1, W2, b2)
                pY = np.array(pY, dtype=np.float64)
                pY_Y = pY - self.T[i*batch_size:(i+1)*batch_size]

                gW2 = (Z.T.dot(pY_Y)/batch_size + reg*W2)
                gb2 = (((pY_Y).sum(axis=0))/batch_size + reg*b2)
                gW1 = ((self.Xtrain[i*batch_size:(i+1)*batch_size].T.dot((pY_Y).dot(W2.T)*(Z > 0)))/batch_size + reg*W1)
                gb1 = ((((pY_Y).dot(W2.T)*(Z > 0)).sum(axis=0))/batch_size + reg*b1)

                mW2 = B1*mW2 + (1 - B1)*gW2
                mb2 = B1*mb2 + (1 - B1)*gb2
                mW1 = B1*mW1 + (1 - B1)*gW1
                mb1 = B1*mb1 + (1 - B1)*gb1

                vW2 = B2*vW2 + (1 - B2)*gW2*gW2
                vb2 = B2*vb2 + (1 - B2)*gb2*gb2
                vW1 = B2*vW1 + (1 - B2)*gW1*gW1
                vb1 = B2*vb1 + (1 - B2)*gb1*gb1

                correctionB1 = 1 - B1**t

                mW2_hat = mW2/correctionB1 
                mb2_hat = mb2/correctionB1 
                mW1_hat = mW1/correctionB1 
                mb1_hat = mb1/correctionB1

                correctionB2 = 1 - B2**t

                vW2_hat = vW2/correctionB2
                vb2_hat = vb2/correctionB2
                vW1_hat = vW1/correctionB2
                vb1_hat = vb1/correctionB2 

                t += 1

                W2 -= learning_rate*mW2_hat/(np.sqrt(vW2_hat) + epsilon)
                b2 -= learning_rate*mb2_hat/(np.sqrt(vb2_hat) + epsilon)
                W1 -= learning_rate*mW1_hat/(np.sqrt(vW1_hat) + epsilon)
                b1 -= learning_rate*mb1_hat/(np.sqrt(vb1_hat) + epsilon)

                # print(i)

        stop_time = time.time()
        full_time = stop_time - start_time
        cost_difference = first_cost - last_cost
        
        print(f'best validation error: {best_validation_error}, cost: {costs[-1]}') 
        print('Cost per second', cost_difference/full_time,)  
        plt.plot(costs, label=label)
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

# np.random.seed(10)
ann = ANN(50, show=False)
# ann.train_full_gd(epochs=100, learning_rate=3e-1)
# ann.train_batch_gd(epochs=100, batch_size=64, learning_rate=2e-2)
# ann.train_batch_gd_with_momentum(epochs=80, batch_size=64, learning_rate=5e-4)
# ann.train_batch_gd_nesterov_momentum(epochs=80, batch_size=64, learning_rate=5e-4, label="nesterov")

learning_rates = [1e-5, 1e-3, 1e-2]
decays = [0.9, 0.99, 0.999]
for learning_rate in learning_rates:
    for decay in decays:
        ann.train_batch_gd_nesterov_momentum_RMS(epochs=150, batch_size=64, decay_rate=decay, learning_rate=learning_rate, label=f'RMS lr: {learning_rate}, decay {decay}')
        # ann.train_batch_gd_nesterov_momentum_RMS(epochs=150, batch_size=64, learning_rate=learning_rate, label=f'Adam lr: {learning_rate}, decay {decay}')
# ann.train_batch_gd_Adam(epochs=150, batch_size=64, learning_rate=1e-2)
# ann.train_sgd_nesterov(epochs=1, learning_rate=6e-2)
# sgd_costs = ann.train_sgd(epochs=1, learning_rate=2e-2)
# ann.train_sgd_AdaGrad(epochs=1, learning_rate=4e-2)
# ann.train_sgd_RMS(epochs=1, learning_rate=5e-4)
plt.legend()
plt.show()
# choice = ''
# while choice.lower() != 'q':
#     choice = input("Number:")
#     ann.predict(int(choice))