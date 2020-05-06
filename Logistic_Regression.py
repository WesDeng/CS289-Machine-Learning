import numpy as np
import scipy.special.expit as sigmoid

# Batch Gradient Descent.


def L2_CrossEntropy(y, w, X, reg):
    A = y * np.log(sigmoid(np.dot(X, w)))
    B = (1 - y) * np.log(1 - sigmoid(np.dot(X, w)))
    return reg * np.linalg.norm(w)/2 - A - B

def BGD(y, w, X, reg, lr, n_iter):
    loss = []
    for _ in range(n_iter):
        linear_model = np.dot(X, w)
        y_predicted = sigmoid(linear_model)

        dw = reg * w - np.dot(X.T, (y_predicted - y))

        w -= lr * dw
        loss.append(L2_CrossEntropy(y, w, X, reg))

    return w, loss



def SGD():
    pass

def


class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000, reg=0.01,
                update='SGD', delta=1):
        """
        Construct a new Logistic Regression instance.

        Inputs:
        - lr: learning rate for BGD and SGD.
        - n_iter: maximum number of interation.
        - reg: regularization parameter.
        - updata: updata rule. (BGD, SGD, step_SGD)
        - delta: hyperparamter for step_SGD
        """

        self.lr = lr
        self.n_iters = n_iters


        self.weights = None
        #self.bias = None

    def fit(self, X, y):
        # Inite the weights.
        N, D = X.shape
        self.weights = np.zeros(D)
        #self.bias = 0

        # Gradient descending combinations.


    def predict(self, X):
        pass
