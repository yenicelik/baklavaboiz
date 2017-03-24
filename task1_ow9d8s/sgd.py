from helper import *

def rms_gradient(X, w, y):
    return np.dot(X.transpose(), np.dot(X, w)) - np.dot(X.transpose(), y)

def weight_update(X, w, y, mu, lr):

    n = X.shape[0]

    weight_term = (1- lr * mu) * w

    return weight_term - lr/n * rms_gradient(X, w, y)

def run_sgd(X_train, y_train, X_cv, y_cv, w, mu, lr, fn_loss, MAX_STEPS=1000):

    train_loss_list = []
    cv_loss_list = []


    for i in range(MAX_STEPS):
        w = weight_update(X_train, w, y_train, mu, lr)
        #The loss should decline.. which it does..
        
        ##Measure loss
        train_predictions = np.dot(X_train, w)
        train_loss = fn_loss(train_predictions, y_train)
        train_loss_list.append(train_loss)

               
        cv_predictions = np.dot(X_cv, w)
        cv_loss = fn_loss(cv_predictions, y_cv)
        cv_loss_list.append(cv_loss)

    return train_loss_list, cv_loss_list, w
       
