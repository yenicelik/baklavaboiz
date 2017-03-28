import numpy as np

def get_train_cross_dataset(X_arr, y_arr, i):
    """ i must be the i'th iteration from k """
    k = len(X_arr)
    X_train = np.concatenate([X_arr[j] for j in range(k) if j != i])
    y_train = np.concatenate([y_arr[j] for j in range(k) if j != i])        
    X_cv = X_arr[i]
    y_cv = y_arr[i]

    return X_train, y_train, X_cv, y_cv

def rms(pred, y):
    out = (1./pred.shape[0]) * np.sum(np.square(pred - y), axis=0)
    return out**0.5


