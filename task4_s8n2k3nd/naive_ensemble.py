#TODO: add l2 regularization to weights!! (we seem to really overfit!)
#TODO: something here is broken; repair this shit


#randomForest   k=73  %71.1
#AdaBoost       k=93  %71.8
#kNN            k=3   %86.6
#SVC        d=1 k=62.1246272545 %90.5
#LDA            k=1   %82.1
#NO QDA
#NO GaussNV

from __future__ import print_function
import numpy as np
import pandas as pd
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
import sys
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import sys
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig

#Import all classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



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
best_accuracy = 0.0
best_k = 1
k=128
#for k in np.arange(40, 50): #1 to 128 with step 4 previously
print("K is now: {}".format(k))
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


def all_predictions2one_prediction(pred_list, size=3):
    all_preds = np.concatenate(pred_list, axis=1)
    out = []
    for i in range(all_preds.shape[0]):
        max_ele = np.bincount(all_preds[i,:].astype('int64'))
        max_ele = np.argmax(max_ele)
        out.append(max_ele)
    return np.asarray(out)



################################
# LABEL THE UNLABALED DATA, AND FEED IT INTO THE NN
################################

uX = train_unlabeled.values[:, :] # shape = (21000, 128)

lda = LDA()
lda.fit(X_train, y_train)
lda_predict = lda.predict(X_cv)
ldaappender = np.reshape(lda_predict, (-1, 1)).astype('int64')

#print("Including SVC")
svc = SVC(degree=1, C=62.1)
svc.fit(X_train, y_train)
svc_predict = svc.predict(X_cv)
svcappender = np.reshape(svc_predict, (-1, 1)).astype('int64')

################################################
# Neural Network  #
################################################

'''
Batch size the higher the better (without going out of mem)
Epoch: How many passes over
'''

# For CV: b_size, epochs, optimizer, layer

b_size = 128
epoch_size = 80 #120 #250
dropout_rate = 0.5

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(BatchNormalization(input_shape=[k])) #must be same dimension as PCA shit
model.add(Dense(700))
#model.add(Dense(700, input_dim=k))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_rate))
model.add(Dense(350))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_rate))
model.add(Dense(125))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_rate))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_list = []
val_list = []

for i in range(epoch_size):
    print("Epoch", i)
    hist = model.fit(X, to_categorical(y), epochs=1, batch_size=b_size)
    train_list.append(hist.history['acc'][0])
    print(hist.history['acc'][0])

    modelpred = model.predict(X_cv, batch_size=b_size)
    uy = all_predictions2one_prediction((svcappender, ldaappender, modelpred))
    acc = accuracy_score(y_cv, uy, normalize=True, sample_weight=None)
    val_list.append(acc)
    print("CV accuracy for NNEnsembleDataInjectionNaive is: ", acc)



lda_predict = lda.predict(X_t)
ldaappender = np.reshape(lda_predict, (-1, 1)).astype('int64')
svc_predict = svc.predict(X_t)
svcappender = np.reshape(svc_predict, (-1, 1)).astype('int64')

acc = model.predict(X_t, batch_size=b_size)
uy = all_predictions2one_prediction((svcappender, ldaappender, modelpred))
acc = accuracy_score(y_cv, uy, normalize=True, sample_weight=None)
print("Final test accuracy is: ", acc[-1])

ind = np.arange(len(train_list))
plt.plot(ind, train_list, 'g')
plt.plot(ind, val_list, 'r--')
savefig('NNEnsembleDataInjectionNaive.png', bbox_inches='tight')
plt.show()


print("Predicting...")
modelpred = model.predict_classes(X_test)
ldaappender = np.reshape(modelpred, (-1, 1)).astype('int64')
lda_predict = lda.predict(X_test)
ldaappender = np.reshape(lda_predict, (-1, 1)).astype('int64')
svc_predict = svc.predict(X_test)
svcappender = np.reshape(svc_predict, (-1, 1)).astype('int64')
y_test = all_predictions2one_prediction((svcappender, ldaappender, modelpred))


################################
# POST PROCESSING
################################

print("Post processing...")
sub_data = np.column_stack((id_values, y_test))
submission = pd.DataFrame(sub_data, columns = ["Id", "y"])

submission.Id = submission.Id.astype(int)
submission.y = submission.y.astype(int)
print("Saving...")
csv_file = submission.to_csv('NNEnsembleDataInjectionNaive.csv', index = False)
print("Done")




