import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import *
from normalEq import *
import time

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

#####################
# PREPROCESSING
data_sample = pd.read_csv('sample.csv')
data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv')

X_data = data_train.values
y_sample = data_sample.values
X_submission = data_test.values[:,1:]

indices = np.arange(X_data.shape[0])
np.random.shuffle(indices)

test_ratio = 0.1
number_of_test = int(X_data.shape[0] * test_ratio) #assuming we want to test out data with 0.1 percent of all the data
X_test = X_data[indices[:number_of_test], :]
X_train = X_data[indices[number_of_test:], :]

# X_train = X_data
# X_test = X_data

#######################
# CROSS VALIDATION AND TRAINING
lam_range = np.logspace(-1, 3, num=1999) #1999

lam_range = np.linspace(250, 280, num=101)

#263.02679919

X = X_train[:,2:]
y = X_train[:,1]

loss_dict = []
lam_dict = []
total_error = 0.0

poly = PolynomialFeatures(degree=3)
reg = linear_model.RidgeCV(alphas=lam_range, fit_intercept=False, store_cv_values=True)
X = poly.fit_transform(X)
#y = poly.fit_transform(y)

reg.fit(X, y)

print(reg.alpha_)

# X_train = poly.fit_transform(X_train)
# predict_ = poly.fit_transform(predict)

# clf = linear_model.LinearRegression()
# clf.fit(X_, vector)
# print clf.predict(predict_)

#########################
# PLOTTING WHERE TO SEARCH NEXT

# print lam_range.shape
# print len(reg.cv_values_)

plt.subplot(3, 1, 1)
plt.title('Absolute training loss')
plt.xlabel('Lambda')
plt.semilogx(reg.cv_values_, 'o')
plt.savefig('foo.png')

# print("Minimal loss", min(loss_dict))
# index = np.argmin(loss_dict)
# indecies = np.argsort(loss_dict)[::-1][-50:][::-1]
# print("Minimizing lam: ", lam_dict[index])
# for i in range(50):
#     print("Loss: " + str(loss_dict[indecies[i]]) + ", lam: " + str(lam_dict[indecies[i]]))


##########################
#TEST SET
print
print("Loss is: ")
X_test = X_test[:,2:]
y_test = X_test[:,1]

X_test = poly.fit_transform(X_test)

y_pred = reg.predict(X_test)
loss = rms(y_pred, y_test)

print(loss)


############################
#Using all the training data

X = X_train[:,2:]
y = X_train[:,1]
X = poly.fit_transform(X)
reg.fit(X, y)

#############################
# calculating and submitting the data

X_submission = poly.fit_transform(X_submission)
y_pred_test = reg.predict(X_submission)

############################
# export
sub_data = np.column_stack((data_test.values[:,0], y_pred_test))
print(sub_data.shape)
print(sub_data)


submission = pd.DataFrame(sub_data, columns = ["Id", "y"])
submission.Id = submission.Id.astype(int)
print(submission)

csv_file = submission.to_csv('final_submission_' + time.strftime("%Y%m%d-%H%M%S") + '-lam:' + str(reg.alpha_) + '.csv')
