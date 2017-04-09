import numpy as np 

def reg_normal_eq(X, y, lam):

	n = X.shape[1]

    #identity = np.identity(n)
    #identity[0,0] = 0 #as was suggested by andrew...

	lhs = lam * np.identity(n) + np.dot(X.transpose(), X)
	rhs = np.dot(X.transpose(), y)


	return np.linalg.solve(lhs, rhs)

def normal_eq(X, y, pinv=True, subtract_mean=False, divide_std=False):
    #Subtract mean
    # if subtract_mean:
    # 	mu = np.mean(X, axis=0)
    # 	X = X - mu
    # 	y = y - mu

    # if divide_std:
    # 	std = np.std(X, axis=0)
    # 	X = X / std
    # 	y = y / std
    
    return np.dot(np.linalg.pinv(X), y)