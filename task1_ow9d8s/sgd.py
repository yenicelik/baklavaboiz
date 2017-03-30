from helper import *

def rms_gradient(X, w, y):
    return np.dot(X.transpose(), np.dot(X, w)) - np.dot(X.transpose(), y)

def weight_update(X, w, y, mu, lr):
    """ check the equations!!!! """
    n = X.shape[0]
    weight_term = (1- lr * mu) * w
    return weight_term - lr/n * rms_gradient(X, w, y)


def run_sgd(X_train, y_train, X_cv, y_cv, w, mu, lr, fn_loss, MAX_STEPS=1000, batch_size=100):
    """ """

    if batch_size > X_train.shape[0]:
        print("batch size bigger than training data!!")

    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    indices = indices[:batch_size] #take random 'batch_size' data samples

    train_loss_list = []
    cv_loss_list = []
    X = X_train[indices, :]
    y = y_train[indices]

    for i in range(MAX_STEPS):
        train_loss, cv_loss, w = sgd_step(X, y, X_cv, y_cv, w, mu, lr, fn_loss)

        if(len(cv_loss_list) > 0 and cv_loss / cv_loss_list[0] > 100):
            #print "Exploding!"
            pass

        train_loss_list.append(train_loss)
        cv_loss_list.append(cv_loss)

    return train_loss_list, cv_loss_list, w


def sgd_step(X_train, y_train, X_cv, y_cv, w, mu, lr, fn_loss, get_loss=True):
    #
    w = weight_update(X_train, w, y_train, mu, lr)
    
    ##Measure loss
    if get_loss:
        train_predictions = np.dot(X_train, w)
        train_loss = fn_loss(train_predictions, y_train)           
        cv_predictions = np.dot(X_cv, w)
        cv_loss = fn_loss(cv_predictions, y_cv)
    
    if get_loss:
        return train_loss, cv_loss, w
    else:
        return [], [], w


       
