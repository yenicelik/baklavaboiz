import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

'''
    Multi-class Classification
    y is 0 - 9
    labeled is indexed by an id
    x1 - x128

'''


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
# We need to add -1 as label to the unlabeled data to classify everything together

# Training Data
X_labeled = train_labeled.values[:, 1:] # shape = (9000, 128)
y_labeled = train_labeled.values[:, 0]
y_unlabeled = np.full((21000,), -1, dtype = int)

X = np.append(X_labeled, train_unlabeled, axis=0) # (9000, 128) - (21000, 128)
y = np.append(y_labeled, y_unlabeled)

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

################################################
# Semi-Supervised Learning with Sklearn #
################################################

# initializing 'best' parameters
best_accuracy = 0.0
best_k = 0

# CROSS VALIDATION

# array of possible choices
iterations = 1
arr = [1] # [1e-5, 1e-6, 1e-4, 1e-3, 1e-2] # np.arange(1, 20, 1, dtype = int)
acc_dict = []
fold_size = int(X_labeled.shape[0] / 5)

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

        # Try changing: kernel and then, gamma, n_neighbors, alpha, max_iter, n_jobs
        lp = LabelSpreading(kernel='rbf', gamma=3, n_neighbors=10, alpha=10, max_iter=10, tol=0.001, n_jobs=10)
        lp.fit(X_train, y_train)
        y_pred = lp.predict(X_cv)

        accuracy += accuracy_score(y_cv, y_pred)

    accuracy /= iterations

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
