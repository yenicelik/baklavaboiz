from __future__ import print_function
from sklearn.lda import LDA
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

import sys
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig


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



#shuffle and create test and cv set
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_train = X[indices[:6000], :]
y_train = y[indices[:6000]]
X_cv =  X[indices[6000:7000], :]
y_cv = y[indices[6000:7000]]
X_t = X[indices[7000:], :]
y_t = y[indices[7000:]]

# Test Data
X_test = test.values[:, :]
id_values = np.arange(30000, 30000+test.shape[0], dtype = int)

# Sanity Check
print("X:")
print(X.shape)
print("y:")
print(y.shape)
print("X_test")
print(X_test.shape)



################################################
# Apply kNN on unlabeled data #
################################################

ks = np.arange(1, 100)
train_list = []
ind_list = []
best_acc = 0.0
best_model = None
for k in ks:
    rfc = LDA(n_components=k)
    rfc.fit(X_train, y_train)
    knn_pred = rfc.predict(X_cv)
    acc = accuracy_score(y_cv, knn_pred)
    if acc > best_acc:
        best_model = rfc
        best_acc = acc
        print("New best LDA model for k: {} and acc: {}".format(k, acc))
    train_list.append(acc)

test_pred = best_model.predict(X_t)
acc = accuracy_score(y_t, test_pred)
print("Final test accuracy is: ", acc)


ind = np.arange(len(train_list))
plt.plot(ind, train_list, 'g')
savefig('LDA.png', bbox_inches='tight')
plt.show()
print("Done!")
sys.exit(0)
