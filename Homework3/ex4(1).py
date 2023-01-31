#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Matern
"""
import random
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import scipy

def prepareData(n=1000):
    """
    Downloads the dataset. Displays some examples.
    Returns the labeled dataset.

    Parameters
    ----------
    n : number of data sample (max 70 000)

    Returns
    -------
    X : Data Matrix
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    """
    mnist_train = torchvision.datasets.MNIST("./data", download=True,)
    mnist_test = torchvision.datasets.MNIST("./data", download=True, train = False)
    
    x,y = mnist_train[50]
    x = np.array(x)
    
    X = []
    y = []
    for x,label in mnist_train:
        X.append(np.array(x))
        y.append(label)

    X_test = []
    y_test = []
    for x,label in mnist_test:
        X_test.append(np.array(x))
        y_test.append(label)
    
    X = np.array(X)
    y = np.array(y)
    X_test= np.array(X_test)
    y_test = np.array(y_test)
    
    sample = random.sample(range(len(X)), n)
    X = np.concatenate((X,X_test))[sample]
    y = np.concatenate((y,y_test))[sample]
    
    imgs = [X[i,:,:] for i in range(10)]
    

    fig=plt.figure(figsize=(8, 5))
    columns = 5
    rows = 2
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(X[i-1,:,:] , cmap ="gray")
        plt.axis('off')
        plt.title(str(y[i-1]))
    plt.show()
    

    return X,y


def featureExtraction(x):
    """
    Applies a feature extraction on a singular image.

    Parameters
    ----------
    x : ndarray
        a numpy array of shape 28x28

    Returns
    -------
    ndarray
        The resulting feature should be one-dimensional (use x.flatten())
    """


    return x.flatten()

def preprocessDataset(X):
    """
    Applies a feature extraction on a dataset

    Parameters
    ----------
    X : ndarray
        Data matrix of size nx28x28
    Returns
    -------
    X_prep : ndarray
        Data matrix of size nxd where d is some dimension of the feature
    """
    
    # TODO: (Optional) You can change this if necessary
    X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    X=X-X.mean(axis=0)
    CO_v=1/(X.shape[0]-1)*X.T.dot(X)
    eig_u,eig_v=scipy.linalg.eigh(CO_v)
    eig_v=eig_v[:,::-1]
    eig_v=eig_v.T[:20]
    X_prep=(eig_v.dot(X.T)).T

    return X_prep

def train(X,y):
    # TODO: Select a classifier from sklearn and train it on the data
    from sklearn.svm import SVC
    model=SVC()
    model.fit(X,y)
    return model


if __name__=="__main__":
    
    # Number of data samples (reduce number during testing if procedure takes too long)
    n = 10000
    X,y = prepareData(n)
    
    # Number of k-folds
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    # Prepare data
    D = preprocessDataset(X)
    
    # K-fold Cross Validation
    i = 0
    test_error = 0
    for train_index, test_index in kf.split(D):
        print("Split: {}".format(i))
        
        X_train, X_test = D[train_index], D[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        model = train(X_train,y_train)
    
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        print("Accuracy Training:", accuracy_score(y_train,y_pred_train))
        print("Accuracy Test:", accuracy_score(y_test,y_pred))
        test_error=test_error+ accuracy_score(y_test,y_pred)
        i= i+1
    
    print("---------------")
    print("Average Test Accuracy: {0:.4f}".format(test_error/n_splits))