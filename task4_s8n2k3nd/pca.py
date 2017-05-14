from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
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
# PRE PROCESSING + REMOVING USING PCA
################################

# Training Data
X = train_labeled.values[:, 1:] # shape = (9000, 128)
y = train_labeled.values[:, 0]

from sklearn.decomposition import PCA
pca = PCA(n_components=90)
pca.fit(X)

sigma = pca.explained_variance_ratio_
print(sigma)
print("Median: {}".format(np.mean(sigma)))
sys.exit(0)





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

