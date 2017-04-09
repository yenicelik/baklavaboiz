import numpy as np
import operator as op
from normalEq import *
import sys

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom
       

def get_train_cross_dataset(X_arr, y_arr, i):
    """ i must be the i'th iteration from k """
    k = len(X_arr)
    X_train = np.concatenate([X_arr[j] for j in range(k) if j != i])
    y_train = np.concatenate([y_arr[j] for j in range(k) if j != i])        
    X_cv = X_arr[i]
    y_cv = y_arr[i]

    return X_train, y_train, X_cv, y_cv

def rms(pred, y):
    out = np.sqrt(((pred - y) ** 2).mean())
    return out


def poly2d_kernel(X):
    """ Will return a new matrix (with new shapes!) X_out that will include permutations of degree=degree as it's elements """
    """ Let's keep this naive, and simple """
    #columns number of the new X_out
    m = X.shape[0]
    n = X.shape[1]

    new_cols = ncr(X.shape[1], 2) + ncr(X.shape[1], 1) + ncr(X.shape[1], 0)

    X_out = np.zeros((X.shape[0], new_cols))

    #Adding the one
    X_out[np.arange(m), 0] = 1

    #Adding the monomial terms
    offset = ncr(X.shape[1], 0)
    for c in range(X.shape[1]):
        X_out[np.arange(m), offset + c] = X[np.arange(m), c] 

    #Adding the binomial terms
    offset = ncr(X.shape[1], 0) + ncr(X.shape[1], 1)
    for c1 in range(X.shape[1]):
        for c2 in range(c1):
            X_out[np.arange(m), offset + c1 + c2] = X[np.arange(m), c1] * X[np.arange(m), c2] #should be elementwise multiplication

    return X_out


def poly3d_kernel(X):
    """ Implement a more naive implementation maybe? (with the counter variable iterating through each value) """
    """ Will return a new matrix (with new shapes!) X_out that will include permutations of degree=degree as it's elements """
    """ Let's keep this naive, and simple """
    #columns number of the new X_out
    m = X.shape[0]
    n = X.shape[1]

    new_cols = ncr(X.shape[1], 3) + ncr(X.shape[1], 2) + ncr(X.shape[1], 1) + ncr(X.shape[1], 0)

    X_out = np.zeros((X.shape[0], new_cols))

    #Adding the one
    X_out[np.arange(m), 0] = 1

    #Adding the monomial terms
    offset = ncr(X.shape[1], 0)
    for c in range(X.shape[1]):
        X_out[np.arange(m), offset + c] = X[np.arange(m), c] 

    #Adding the binomial terms
    offset = ncr(X.shape[1], 0) + ncr(X.shape[1], 1)
    for c1 in range(X.shape[1]):
        for c2 in range(c1):
            X_out[np.arange(m), offset + c1 + c2] = X[np.arange(m), c1] * X[np.arange(m), c2] #should be elementwise multiplication


    #Adding the binomial terms
    offset = ncr(X.shape[1], 2) + ncr(X.shape[1], 1) + ncr(X.shape[1], 0)
    for c1 in range(X.shape[1]):
        for c2 in range(c1):
            for c3 in range(c2):
                X_out[np.arange(m), offset + c1 + c2 + c3] = X[np.arange(m), c1] * X[np.arange(m), c2] * X[np.arange(m), c3] #should be elementwise multiplication

    return X_out

