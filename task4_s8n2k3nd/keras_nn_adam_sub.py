import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score


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
y = to_categorical(train_labeled.values[:, 0])

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
# Neural Network  #
################################################

'''
Batch size the higher the better (without going out of mem)
Epoch: How many passes over
'''

# For CV: b_size, epochs, optimizer, layer

b_size = 128
epoch_size = 250
dropout_rate = 0.5

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(700, activation='relu', input_dim=128))
model.add(Dropout(dropout_rate))
model.add(Dense(350, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(125, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, epochs=epoch_size, batch_size=b_size)
y_test = model.predict_classes(X_test)


################################
# POST PROCESSING
################################

sub_data = np.column_stack((id_values, y_test))
submission = pd.DataFrame(sub_data, columns = ["Id", "y"])

submission.Id = submission.Id.astype(int)
submission.y = submission.y.astype(int)
csv_file = submission.to_csv('final_submission.csv', index = False)
