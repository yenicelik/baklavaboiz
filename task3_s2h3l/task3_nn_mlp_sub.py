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

id_values = np.arange(train.shape[0], train.shape[0]+8137, dtype = int)


################################################
# Neural Network with Multi Layered Perceptron #
################################################

# MLP trains using Backpropagation -> Uses SGD and the gradients are calculated using Backpropagation.
# For classification, it minimizes the Cross-Entropy loss function

mlp = MLPClassifier(solver='adam', alpha=0.0001, max_iter=2000, hidden_layer_sizes=(700,), tol=1e-4, random_state=1)
mlp.fit(X, y)
y_test = mlp.predict(X_test)

################################
# POST PROCESSING
################################

sub_data = np.column_stack((id_values, y_test))
submission = pd.DataFrame(sub_data, columns = ["Id", "y"])

submission.Id = submission.Id.astype(int)
submission.y = submission.y.astype(int)
csv_file = submission.to_csv('final_submission.csv', index = False)
