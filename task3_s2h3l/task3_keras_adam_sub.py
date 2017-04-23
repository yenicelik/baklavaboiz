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
# Neural Network  #
################################################

'''
b_size := Batch Size
Epoch: How many passes over the dataset
'''

b_size = 128

model = Sequential()
# Dense(700) is a fully-connected layer with 700 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 100-dimensional vectors.
model.add(Dense(700, activation='relu', input_dim=100))
model.add(Dropout(0.5))
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=b_size)
y_test = model.predict_classes(X_test)


################################
# POST PROCESSING
################################

sub_data = np.column_stack((id_values, y_test))
submission = pd.DataFrame(sub_data, columns = ["Id", "y"])

submission.Id = submission.Id.astype(int)
submission.y = submission.y.astype(int)
csv_file = submission.to_csv('final_submission.csv', index = False)
