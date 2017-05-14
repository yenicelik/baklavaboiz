from __future__ import print_function
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
import sys
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Reshape

from keras.layers.normalization import BatchNormalization

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
# Neural Network  #
################################################

'''
Batch size the higher the better (without going out of mem)
Epoch: How many passes over
'''

# For CV: b_size, epochs, optimizer, layer

b_size = 128
epoch_size = 120 #120 #250
dropout_rate = 0.5

model = Sequential()
model.add(Reshape((128, 1), input_shape=[128]))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


train_list = []
val_list = []

for i in range(epoch_size):
    hist = model.fit(X, y, epochs=1, batch_size=b_size)
    acc = model.evaluate(X_cv, y_cv, batch_size=b_size)
    train_list.append(hist.history['acc'][0])
    print(hist.history['acc'][0])
    val_list.append(acc[-1])
    print("CV accuracy is: ", acc[-1])

acc = model.evaluate(X_t, y_t, batch_size=b_size)
print("Final test accuracy is: ", acc[-1])

ind = np.arange(len(train_list))
plt.plot(ind, train_list, 'g')
plt.plot(ind, val_list, 'r--')
savefig('nonBN.png', bbox_inches='tight')
plt.show()


print("Predicting...")
y_test = model.predict_classes(X_test)


################################
# POST PROCESSING
################################

print("Post processing...")
sub_data = np.column_stack((id_values, y_test))
submission = pd.DataFrame(sub_data, columns = ["Id", "y"])

submission.Id = submission.Id.astype(int)
submission.y = submission.y.astype(int)
print("Saving...")
csv_file = submission.to_csv('final_submission-nonBN.csv', index = False)
print("Done")
