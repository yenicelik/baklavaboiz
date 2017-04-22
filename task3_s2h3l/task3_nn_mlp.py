import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

################################
# IMPORT DATA
################################

train = pd.read_hdf('train.h5', 'train')
test = pd.read_hdf('test.h5', 'test')
print(train.shape)
print(test.shape)

################################
# PRE PROCESSING
################################

# Training Data
X = train.values[:, 1:]
y = train.values[:, 0]

print(X.shape)
print(y.shape)

# Test Data
X_test = test.values[:, :]

print("X_test")
print(X_test.shape)

id_values = np.arange(X_test.shape[0], X_test.shape[0]+8137, dtype = int)


################################################
# Neural Network with Multi Layered Perceptron #
################################################

# MLP trains using Backpropagation -> Uses SGD and the gradients are calculated using Backpropagation.
# For classification, it minimizes the Cross-Entropy loss function

# initializing 'best' parameters
best_accuracy = 0.0
best_k = 0

# CROSS VALIDATION

# array of possible choices
iterations = 5
arr = [1] # [1e-5, 1e-6, 1e-4, 1e-3, 1e-2] # np.arange(1, 20, 1, dtype = int)
acc_dict = []
fold_size = int(X.shape[0] / 5)

for k in arr:
    accuracy = 0.0
    print("Using: ", k)

    for i in range(iterations): # take the average of 5 iterations

        # creating training and cross validation data
        indicies = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indicies)

        X_train = X[indicies[fold_size:], :]
        y_train = y[indicies[fold_size:]]
        X_cv = X[indicies[:fold_size], :]
        y_cv = y[indicies[:fold_size]]

        # Which parameters to optimize? Hidden Layer?
        # alpha is regularization parameter -> avoiding overfitting by penalizing weights
        # Use LBFGS as solver: "For small datasets, however, ‘lbfgs’ can converge faster and perform better."
        # But you could also try to vary it
        # try varying the hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
        mlp = MLPClassifier(solver='adam', alpha=0.0001, max_iter=2000, hidden_layer_sizes=(100,100,50), tol=1e-4, random_state=1)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_cv)

        accuracy += accuracy_score(y_cv, y_pred)

    accuracy /= iterations

    if best_accuracy < accuracy:
        best_k = k
        best_accuracy = accuracy

    acc_dict.append(accuracy)

print("Best accuracy: ", best_accuracy)
print("Best parameter: ", best_k)


'''
mlp = MLPClassifier();
mlp.fit(X, y)
y_test = mlp.predict(X_test)

################################
# POST PROCESSING
################################

sub_data = np.column_stack((test.values[:,0], y_test))
submission = pd.DataFrame(sub_data, columns = ["Id", "y"])

submission.Id = submission.Id.astype(int)
submission.y = submission.y.astype(int)
print(submission)
csv_file = submission.to_csv('final_submission.csv', index = False)
'''
