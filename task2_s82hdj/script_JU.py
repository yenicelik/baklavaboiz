import pandas
import numpy as np
import random
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plotter


training_data = pandas.read_csv('train.csv')
test_data = pandas.read_csv('test.csv')


X = training_data.values[:,2:]
y = training_data.values[:,1]
X_submission = test_data.values[:,1:]

poly = PolynomialFeatures(3, interaction_only = True)
X = poly.fit_transform(X)
X_submission = poly.fit_transform(X_submission)

vars = np.arange(6)

max_score = 0
max_var = vars[0]

plotter.xlabel("var")
plotter.ylabel("avg_score")

for var in vars:

    clf = DecisionTreeClassifier(random_state = var, criterion='entropy')

    clf.fit(X, y)
    
    
    scores = cross_val_score(clf, X, y, cv=6)
    avg_score = np.sum(scores)/len(scores)

    print avg_score

    if avg_score > max_score:
        max_score = avg_score
        max_var = var
    

    '''
    prediction = clf.predict(X_submission)

    submission_data = np.column_stack((test_data.values[:,0], prediction))
    submission = pandas.DataFrame(submission_data, columns=["Id", "y"])
    submission.Id = submission.Id.astype(int)
    submission.y = submission.y.astype(int)

    submission.to_csv('submission_JU.csv', index=False)
    '''


print max_score
print max_var

# 2
plotter.scatter(X[:, 14], y, color="blue")
#plotter.show()