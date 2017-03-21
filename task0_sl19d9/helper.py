import numpy as np 


def train_test_split(X, y, fold_size):
	""" """

	indices = np.arange(X.shape[0])
	np.random.shuffle(indices)
	
	#extracting the 
	X_train = X[indices[fold_size:], :]
	y_train = y[indices[fold_size:]]
	X_cv = X[indices[:fold_size], :]
	y_cv = y[indices[:fold_size]]

	return X_train, X_cv, y_train, y_cv


def normalEq(X, y, pinv=True, subtract_mean=False, divide_std=False):
    #Subtract mean
    if subtract_mean:
    	mu = np.mean(X, axis=0)
    	X = X - mu
    	y = y - mu

    if divide_std:
    	std = np.std(X, axis=0)
    	X = X / std
    	y = y / std
    
    if pinv:
    	return np.dot(np.linalg.pinv(X), y)
    else:
    	lhs = np.dot(X.transpose(), X)
    	lhsinv = np.linalg.inv(lhs)
    	rhs = np.dot(X.transpose(), y)
    	return np.dot( lhsinv, rhs)

def rms(pred, y):
    out = (1./pred.shape[0]) * np.sum(np.square(pred - y), axis=0)
    return out**0.5

def numerical_gradient(f, x):
	""" Helper function to test if our derivative is accurate """
	fx = f(x)
	grad = np.zeros(x.shape)
	h = 1e-5 #just testing, so good enough

	# iterate over all indexes in x
  	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  	while not it.finished:

	    # evaluate function at x+h
	    ix = it.multi_index
	    old_value = x[ix]
	    x[ix] = old_value + h # increment by h
	    fxh = f(x) # evalute f(x + h)
	    x[ix] = old_value # restore to previous value (very important!)

	    # compute the partial derivative
	    grad[ix] = (fxh - fx) / h # the slope
    	it.iternext() # step to next dimension

	return grad

def rms_gradient():
	pass
