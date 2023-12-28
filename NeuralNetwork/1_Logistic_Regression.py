from __future__ import print_function, division
from builtins import range
# sudo pip install -U future

# Note: Run this script from the current folder.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def get_clouds():
    Nclass = 500
    D = 2
    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    return X, Y

def get_spiral():
    # Idea: radius -> low...high
    #           (don't start at 0, otherwise points will be "mushed" at origin)
    #       angle = low...high proportional to radius
    #               [0, 2pi/6, 4pi/6, ..., 10pi/6] --> [pi/2, pi/3 + pi/2, ...,]
    # x = rcos(theta), y = rsin(theta) as usual

    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi*i / 3.0
        end_angle = start_angle + np.pi / 2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points

    # convert into cartesian coordinates
    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])

    # inputs
    X = np.empty((600, 2))
    X[:,0] = x1.flatten()
    X[:,1] = x2.flatten()

    # add noise
    X += np.random.randn(600, 2)*0.5

    # targets
    Y = np.array([0]*100 + [1]*100 + [0]*100 + [1]*100 + [0]*100 + [1]*100)
    return X, Y

def get_transformed_data():
    print("Reading in and transforming data...")

    if not os.path.exists('/large_files/train.csv'):
        print('Looking for /large_files/train.csv')
        print('You have not downloaded the data and/or not placed the files in the correct location.')
        print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
        print('Place train.csv in the folder large_files adjacent to the class folder')
        exit()

    df = pd.read_csv('/large_files/train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data) #important for train-test-split -> to randomize labels

    X = data[:, 1:]
    Y = data[:, 0].astype(np.int32)

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:]
    Ytest  = Y[-1000:]

    # center the data
    mu = Xtrain.mean(axis=0)
    Xtrain = Xtrain - mu
    Xtest  = Xtest - mu

    # transform the data
    pca = PCA()
    Ztrain = pca.fit_transform(Xtrain)
    Ztest  = pca.transform(Xtest)

    plot_cumulative_variance(pca)

    # take first 300 cols of Z
    Ztrain = Ztrain[:, :300]
    Ztest = Ztest[:, :300]

    # normalize Z
    mu = Ztrain.mean(axis=0)
    std = Ztrain.std(axis=0)
    Ztrain = (Ztrain - mu) / std
    Ztest = (Ztest - mu) / std

    return Ztrain, Ztest, Ytrain, Ytest

# function returns training & test dataset
def get_normalized_data():
    print("Reading in and transforming data...")
    #check if file exists
    if not os.path.exists('/content/large_files/train.csv'):
        print('Looking for /large_files/train.csv')
        print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
        exit() #terminate script if file is not present

    df = pd.read_csv('/content/large_files/train.csv')
    data = df.values.astype(np.float32) #data already flattened into 1D array
    np.random.shuffle(data)  #important for train-test-split -> to randomize labels

    #split data into i/p & targets
    X = data[:, 1:] #data: rest of the columns
    Y = data[:, 0] #target: 1st column

    #split data into train & test
    Xtrain = X[:-1000] #start: 1000
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:] #last 1000 (validation set)
    Ytest  = Y[-1000:]

    #normalize the data (standardization: every column: u=0, var=1)
    mu = Xtrain.mean(axis=0) #mean along the rows
    std = Xtrain.std(axis=0) #std along the rows
    #replace '0' with '1'
    np.place(std, std == 0, 1)
    #standardization/normalization of both 'train' & 'test' set
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std

    return Xtrain, Xtest, Ytrain, Ytest

def plot_cumulative_variance(pca):
    P = []
    for p in pca.explained_variance_ratio_:
        if len(P) == 0:
            P.append(p)
        else:
            P.append(p + P[-1])
    plt.plot(P)
    plt.show()
    return P

#Takes training set, weight & bias -> returns arrays of prob for each sample 'N'
def forward(X, W, b):
    # softmax
    a = X.dot(W) + b #Linear Tx of 'W' & 'b'
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True) #softmax -> each class is a prob, exp to make all val. pos for prob; calc. for each sample 'N'
    return y

#Takes prob(y) -> returns class
def predict(p_y):
    return np.argmax(p_y, axis=1) #predicted class will  have the highest prob.

#Takes prob(y) & y_test  -> returns proportion of misclassified error
def error_rate(p_y, t):
    prediction = predict(p_y) #predicted class label (0-9), t: True label
    return np.mean(prediction != t) #gives proportion of incorrect labels

#Takes arrays of prob & indicator matrix -> returns cross entropy loss
def cost(p_y, t):
    tot = t * np.log(p_y) #element wise matrix multiplication; t: one-hot coded Target; p_y(matrix) : prob of y_train | x_train
    return -tot.sum()/len(t) #model will try to reduce this cost adjusting, 'W' & 'B'

# Takes 'target label', 'predicted label', & 'training-set' -> returns ΔW
# Gradient Ascent:
def gradW(t, y, X):
    return X.T.dot(t - y)

# Takes 'target label', 'predicted label' -> returns ΔB
def gradb(t, y):
    return (t - y).sum(axis=0)

#Takes training set -> returns indicator matrix with '1' corresponding to label
def y2indicator(y):
    N = len(y) # No. of training/ test samples
    y = y.astype(np.int32)
    k = y.max()+1
    ind = np.zeros((N, k)) #matrix  of size (#of samples, #of classes)
    for i in range(N):
        ind[i, y[i]] = 1 #indicator matrix have 1 corresponding to class value for each sample
    return ind

def benchmark_full():
  #load data
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    print("Performing logistic regression...")
    # lr = LogisticRegression(solver='lbfgs')
    # convert Ytrain and Ytest to (N x K) matrices of indicator variables
    N, D = Xtrain.shape #extract N: #of samples, D: # of features
    Ytrain_ind = y2indicator(Ytrain) #
    Ytest_ind = y2indicator(Ytest)
    #initialize logistic regression parameters
    k=10
    W = np.random.randn(D, k) / np.sqrt(D) # weight matrix of logistic regression -> connecting "features" of i/p vector to "class" of o/p vector;
                                           # normalization is essential to prevent gradients from vanishing or exploding
    b = np.zeros(k)
    #plotting parameters
    train_losses = [] #train loss
    LLtest = [] #test loss
    CRtest = [] #test classification errors

    #trial and error: to find good learning rate & regularization penalty
    # reg = 1
    # learning rate 0.0001 is too high, 0.00005 is also too high
    # 0.00003 / 2000 iterations => 0.363 error, -7630 cost
    # 0.00004 / 1000 iterations => 0.295 error, -7902 cost
    # 0.00004 / 2000 iterations => 0.321 error, -7528 cost

    # reg = 0.1, still around 0.31 error
    # reg = 0.01, still around 0.31 error

    # assign initial val for 1st trial
    lr = 0.00004  #learning rate
    reg = 0.01    #regularization penalty
    n_iter=50
    for i in range(n_iter):
        p_y = forward(Xtrain, W, b)
        ll = cost(p_y, Ytrain_ind)
        train_losses.append(ll)

        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)

        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)

        #gradient ascent -> numerically equivalent to gradient descent
        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W) #update 'W' based on penalty, and 'regularization' to prevent overfitting
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b) #update 'B' based on penalty, and 'regularization' to prevent overfitting
        if (i+1) % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i, ll))
            print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    iters = range(len(train_losses))
    plt.plot(iters, train_losses, label='train loss')
    plt.plot(iters, LLtest, label='test loss')    
    plt.title('Benchmark Single Layer NN')
    plt.legend(loc="upper right")
    plt.xlabel('# of iteration')
    plt.ylabel('average loss/sample')
    plt.show()
    plt.plot(CRtest)
    plt.show()

def benchmark_pca():
    Xtrain, Xtest, Ytrain, Ytest = get_transformed_data()
    print("Performing logistic regression...")

    N, D = Xtrain.shape
    k=Ytrain.astype(np.int32).max()+1
    Ytrain_ind = np.zeros((N, k))
    for i in range(N):
        Ytrain_ind[i, Ytrain[i]] = 1

    Ntest = len(Ytest)
    Ytest_ind = np.zeros((Ntest, 10))
    for i in range(Ntest):
        Ytest_ind[i, Ytest[i]] = 1

    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    train_losses = []
    LLtest = []
    CRtest = []

    # D = 300 -> error = 0.07
    lr = 0.0001
    reg = 0.01
    for i in range(200):
        p_y = forward(Xtrain, W, b)
        # print "p_y:", p_y
        train_loss = cost(p_y, Ytrain_ind)
        train_losses.append(train_loss)

        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)

        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) + reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) + reg*b)
        if i % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i, train_loss))
            print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    iters = range(len(train_losses))
    plt.plot(iters, train_losses, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.title('Benchmark PCA')
    plt.show()

if __name__ == '__main__':
    # benchmark_pca()
    benchmark_full()
