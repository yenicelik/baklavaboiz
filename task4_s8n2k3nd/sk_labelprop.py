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

##############################################################
# Semi-Supervised Learning with Sklearn and LabelPropagation #
##############################################################

# CROSS VALIDATION
kfolds = 5

#do crossvalidation setup
kf = KFold(n_splits=kfolds, shuffle= True, random_state = None)
kf.get_n_splits(X)

sum_of_accuracy_score = 0
iteration = 0

# we now go the indices and now take every possibility of training on k-1 and testing on 1
for train_index, test_index in kf.split(X_labeled):
    iteration += 1
    print("We are in iteration " + repr(iteration))

    X_train, X_cv = X[train_index], X[test_index]
    y_train, y_cv = y[train_index], y[test_index]

    label_prop_model = LabelPropagation(kernel='rbf',gamma=5,alpha=k)
    print("Start calculation...")
    label_prop_model.fit(X_train, y_train)

    #lets now apply it on the validation set.
    y_pred = label_prop_model.predict(X_cv)
    acc = accuracy_score(y_cv, y_pred)
    sum_of_accuracy_score += acc
    print("The score for this iteration is " + repr(acc))

print("The average accuracy score is " + repr(sum_of_accuracy_score/kfolds))

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
