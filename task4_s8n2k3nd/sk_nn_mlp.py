import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import sys



################################
# IMPORT DATA
################################

train_labeled = pd.read_hdf('train_labeled.h5', 'train')
print(train_labeled.shape)  # shape: y + 128 x = 129 columns
train_unlabeled = pd.read_hdf('train_unlabeled.h5', 'train')
print(train_unlabeled.shape) # shape: 128 x columns
test = pd.read_hdf('test.h5', 'test')
print(test.shape) # shape: 128 columns (we have no ID column, which we need to add to our sub file later)


################################
# PRE PROCESSING
################################

# Training Data
X = train_labeled.values[:, 1:] # shape = (9000, 128)
y = train_labeled.values[:, 0]

# Test Data
X_test = test
id_values = np.arange(30000, 30000+test.shape[0], dtype = int)

# Sanity Check
print("X:")
print(X.shape)
print("y:")
print(y.shape)
print("X_test")
print(X_test.shape)

sys.exit(0)

'''
# FEATURE EXPANSION
poly = PolynomialFeatures(degree=2, include_bias=True)
poly.fit_transform(X)


# SCALING
# Multi-layer Perceptron is sensitive to feature scaling
# -> StandardScaler transforms data: zero mean and unit variance
scaler = StandardScaler()
# Scale training Data
scaler.fit(X)
X = scaler.transform(X)
# apply same transformation to test data
X_test = scaler.transform(X_test)
'''

################################################
# Neural Network with Multi Layered Perceptron #
################################################

best_accuracy = 0.0
best_k = 0

# CROSS VALIDATION

iterations = 50
arr = np.arange(100, 500, 200, dtype = int) # [100] # [1e-5, 1e-6, 1e-4, 1e-3, 1e-2] # np.arange(1, 20, 1, dtype = int)
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


        mlp = MLPClassifier(solver='adam', alpha=0.01, hidden_layer_sizes=(600,k), max_iter=2000, tol=1e-4, random_state=1)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_cv)

        accuracy += accuracy_score(y_cv, y_pred)

    accuracy /= iterations
    print("Accuracy: ", accuracy)
    
    if best_accuracy < accuracy:
        best_k = k
        best_accuracy = accuracy

    acc_dict.append(accuracy)

print("Best accuracy: ", best_accuracy)
print("Best parameter: ", best_k)



'''
################################
# POST PROCESSING
################################

sub_data = np.column_stack((id_values, y_test))
submission = pd.DataFrame(sub_data, columns = ["Id", "y"])

submission.Id = submission.Id.astype(int)
submission.y = submission.y.astype(int)
csv_file = submission.to_csv('final_submission.csv', index = False)
'''
