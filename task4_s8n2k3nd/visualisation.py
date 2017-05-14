import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plotter
from mpl_toolkits.mplot3d import Axes3D

train_labeled = pd.read_hdf('train_labeled.h5', 'train')
train_unlabeled = pd.read_hdf('train_unlabeled.h5', 'train')

X = train_labeled.values[:, 1:]
y = train_labeled.values[:, 0]
N = X.shape[0]

tsne2d = TSNE(n_components = 2)

# class = 0 makes a red dot, class = 1 an orange dot and so on...
colors = ["red", "orange", "yellow", "yellowgreen", "green", "turquoise", "lightblue", "darkblue", "violet", "black"]
# this matrix was made for reusability, but doesn't serve a big purpose now
# color_matrix[i] jsut gives the color for the i-th point
color_matrix = []
for i in xrange(N):
    color_matrix.append(colors[int(y[i])])

X_2D = tsne2d.fit_transform(X)

# matplotlib gives you mulitple canvases with pyplot.figure(id)
plt_2d_figure = plotter.figure(0)
# but to actually get the canvas you have to do this for some reason
plt_2d = plt_2d_figure.add_subplot(111)

plt_2d.scatter(X_2D[:,0], X_2D[:,1], color = color_matrix)


# uncomment this to unlock the 3d plot
# CAREFUL:
# it's as underwhelming as the 2d plot
'''
tsne3d = TSNE(n_components = 3)

plt_3d_figure = plotter.figure(1)
plt_3d = plt_3d_figure.add_subplot(111, projection='3d')
X_3D = tsne3d.fit_transform(X)
plt_3d.scatter(X_3D[:,0], X_3D[:,1], X_3D[:,2], color = color_matrix)

'''

plotter.show()