import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
    
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

################################
# IMPORT DATA
################################

train = pd.read_csv('train.csv') # better be in the correct directory!
test = pd.read_csv('test.csv')

X = train.values[:, 2:]
y = train.values[:, 1]

# print(train.values[:, 0])
# print(X.shape)
# print(y.shape)

X_test = test.values[:, 1:]
X_id = test.values[:, 0]
print(X_id)

print(X_test.shape)
print(X_id.shape)


################################
# CHOOSING PARAMETER THROUGH CV
################################
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# #initializing 'best' parameters
# best_accuracy = 0.0
# best_k = 0

# #array of possible choices
# iterations = 500 #we want to make sure that one parameter was not chosen 'by chance'
# k_arr = np.arange(1, 100, dtype=int) #possible do an arange?
# acc_dict = []
# fold_size = X.shape[0] * 0.2

# for k in k_arr:
# 	accuracy = 0.0

# 	for i in range(iterations): #take the average of 5 iterations
		
# 		#creating training and cross validation data
# 		indecies = np.arange(X.shape[0], dtype=int)
# 		np.random.shuffle(indecies)

# 		X_train = X[indecies[fold_size:], :]
# 		y_train = y[indecies[fold_size:]]
# 		X_cv = X[indecies[:fold_size], :]
# 		y_cv = y[indecies[:fold_size]]

# 		knn = KNeighborsClassifier(n_neighbors=k) #any more parameters to set?
# 		knn.fit(X_train, y_train)
# 		y_pred = knn.predict(X_cv)
# 		accuracy += accuracy_score(y_cv, y_pred)

# 	accuracy /= iterations

# 	if best_accuracy < accuracy:
# 		best_k = k
# 		best_accuracy = accuracy

# 	best_accuracy /= iterations
# 	acc_dict.append(accuracy)

# print("Best accuracy: ", best_accuracy)
# print("Best k: ", best_k) #15 etc. #12 #14 it is

# # Displaying results in matplotlib
# plt.plot(k_arr, acc_dict, 'ro')
# plt.show()


################################
# SUBMITTING AND TESTING STUFF
################################
knn = KNeighborsClassifier(n_neighbors=13) #any more parameters to set?
knn.fit(X, y)
y_test = knn.predict(X_test)

sub_data = np.column_stack((test.values[:,0], y_test))
submission = pd.DataFrame(sub_data, columns = ["Id", "y"])

submission.Id = submission.Id.astype(int)
submission.y = submission.y.astype(int)
csv_file = submission.to_csv('final_submission.csv', index = False)








