#TODO: add l2 regularization to weights!! (we seem to really overfit!)

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
k=128#44
#for k in np.arange(40, 50): #1 to 128 with step 4 previously
print("K is now: {}".format(k))
X = train_labeled.values[:, 1:] # shape = (9000, 128)
y = train_labeled.values[:, 0]

from sklearn.decomposition import PCA

#pca = PCA(n_components=k)
#X = pca.fit_transform(X)


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
# Build naive ensemble model
################################################

#Random Forest k=73  %71.1
if False:
    #print("Including RandomForest")
    rfc = RandomForestClassifier(n_estimators=73)
    rfc.fit(X_train, y_train) #to_categorical
    rfc_predict = rfc.predict(X_cv)
    rfcappender = np.reshape(rfc_predict, (-1, 1)).astype('int64')
    acc = accuracy_score(y_cv, rfc_predict) #To see how much we would get if we apply this isolatedly

#AdaBoost k=93 %71.8
if False:
    #print("Including AdaBoost")
    abc = AdaBoostClassifier(n_estimators=93)
    abc.fit(X_train, y_train)
    abc_predict = abc.predict(X_cv)
    abcappender = np.reshape(abc_predict, (-1, 1)).astype('int64')
    acc = accuracy_score(y_cv, abc_predict) #To see how much we would get if we apply this isolatedly

#kNN            k=3   %86.6
if False:
    #print("Including KNN")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    knn_predict = knn.predict(X_cv)
    knnappender = np.reshape(knn_predict, (-1, 1)).astype('int64')
    acc = accuracy_score(y_cv, knn_predict) #To see how much we would get if we apply this isolatedly

#SVC        d=1 k=62.1246272545 %90.5
if False:
    #print("Including SVC")
    svc = SVC(degree=1, C=62.1)
    svc.fit(X_train, y_train)
    svc_predict = svc.predict(X_cv)
    svcappender = np.reshape(svc_predict, (-1, 1)).astype('int64')
    acc = accuracy_score(y_cv, svc_predict) #To see how much we would get if we apply this isolatedly

#LDA            k=1   %82.1
if False:
    #print("Including LDA")
    lda = LDA()
    lda.fit(X_train, y_train)
    lda_predict = lda.predict(X_cv)
    ldaappender = np.reshape(lda_predict, (-1, 1)).astype('int64')
    acc = accuracy_score(y_cv, lda_predict) #To see how much we would get if we apply this isolatedly



def all_predictions2one_prediction(pred_list, size=3):
    all_preds = np.concatenate(pred_list, axis=1)
    out = []
    for i in range(all_preds.shape[0]):
        max_ele = np.bincount(all_preds[i,:])
        max_ele = np.argmax(max_ele)
        out.append(max_ele)
    return np.asarray(out)


best_config = 0000
best_acc = 0.0
i = 0
while i < 0: #2**6:
    all_predictions = []
    if (i % 2) >= 1: #LDA
        all_predictions.append(ldaappender)
    if (i % 4) >= 2: #include RandomForest
        all_predictions.append(rfcappender)

    if (i % 8) >= 4: #include AdaBoost
        all_predictions.append(abcappender)

    if (i % 16) >= 8: #kNN
        all_predictions.append(knnappender)

    if (i % 32) >= 16: #SVC
        all_predictions.append(svcappender)

    i += 1

    if len(all_predictions) == 0:
        continue

    ensemble_prediction = all_predictions2one_prediction(all_predictions)
    acc = accuracy_score(y_cv, ensemble_prediction)
    if acc > best_acc:
        best_config = i
        best_acc = acc
        print("Best config: i {} with acc {} and dimensions {}".format(i, acc, k))

    #if best_acc < best_accuracy:
    #    best_k = k
    #    best_accuracy = best_acc

#print("Got best accuracy with: {} at {}".format(best_k, best_accuracy))



################################
# LABEL THE UNLABALED DATA, AND FEED IT INTO THE NN
################################

uX = train_unlabeled.values[:, :] # shape = (21000, 128)
#uX = pca.fit_transform(uX)

lda = LDA()
lda.fit(X_train, y_train)
lda_predict = lda.predict(uX)
ldaappender = np.reshape(lda_predict, (-1, 1)).astype('int64')

#print("Including SVC")
svc = SVC(degree=1, C=62.1)
svc.fit(X_train, y_train)
svc_predict = svc.predict(uX)
svcappender = np.reshape(svc_predict, (-1, 1)).astype('int64')





print("svcappender is {}".format(svcappender.shape))
print("ldaappender shape is {}".format(ldaappender.shape))
uy = all_predictions2one_prediction((svcappender, ldaappender))

print("uX shape is {}".format(uX.shape))
print("uy shape is {}".format(uy.shape))

#only accept the ones where both predictors agree upon



################################################
# Neural Network  #
################################################

'''
Batch size the higher the better (without going out of mem)
Epoch: How many passes over
'''

# For CV: b_size, epochs, optimizer, layer

b_size = 128
epoch_size = 70 #120 #250
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


print("X shape is {}".format(X.shape))
print("y shape is {}".format(y.shape))

X = np.concatenate((X, uX), axis=0) #real values should have more weight
y = np.concatenate((y, uy), axis=0) #real values should have more weight

print("X shape is {}".format(X.shape))
print("y shape is {}".format(y.shape))

train_list = []
val_list = []

for i in range(epoch_size):
    print("Epoch {}".format(i))
    hist = model.fit(X, to_categorical(y), epochs=1, batch_size=b_size)
    #X_cv = pca.fit_transform(X_cv)
    acc = model.evaluate(X_cv, to_categorical(y_cv), batch_size=b_size)
    train_list.append(hist.history['acc'][0])
    print(hist.history['acc'][0])
    val_list.append(acc[-1])
    print("CV accuracy for NNEnsembleDataInjectionBNsX is: ", acc[-1])

#X_t = pca.fit_transform(X_t)
acc = model.evaluate(X_t, to_categorical(y_t), batch_size=b_size)
print("Final test accuracy is: ", acc[-1])

ind = np.arange(len(train_list))
plt.plot(ind, train_list, 'g')
plt.plot(ind, val_list, 'r--')
savefig('NNEnsembleDataInjectionBNsX.png', bbox_inches='tight')
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
csv_file = submission.to_csv('NNEnsembleDataInjectionBNsX.csv', index = False)
print("Done")




