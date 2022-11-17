#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: 

Summary of (part of the) functions considered in the course
    
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, linear_model, svm, cluster
from sklearn import metrics, decomposition


# load digits
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Plot images of the digits
n_img_per_row = 6
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = digits.images[i * n_img_per_row + j]

plt.figure(1)
plt.clf()
plt.imshow(img, cmap=plt.cm.gray_r)
plt.xticks([])
plt.yticks([])
plt.title('Some digits in the dataset')
plt.show()

#%%
# Clustering 
kmn = cluster.KMeans(init='k-means++', n_clusters=10, n_init=10)
kmn.fit(X)
centers = kmn.cluster_centers_
labels = kmn.labels_
score = kmn.inertia_





# Dimensionality reduction using PCA
pca = decomposition.PCA(n_components=None)
pca.fit(X)

plt.figure(2)
plt.clf()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('components')
plt.ylabel('cumulated variance ratio')
plt.show()

pca.set_params(n_components=2)
Xred = pca.fit_transform(X)

Cred = pca.fit_transform(centers)

plt.figure(3)
plt.clf()
plt.set_cmap(plt.cm.jet)
classes=range(10)
plt.scatter(Xred[:,0],Xred[:,1],c=y,s=6)
for i in range(10):
    plt.scatter([],[],label=i)
    
plt.plot(Cred[:,0],Cred[:,1],'d',markersize=10,markeredgecolor='k',markerfacecolor='w',label='centers')
    
plt.xlim(tuple(map(sum,zip((0,10),plt.xlim()))))
plt.legend(loc='upper right')
plt.show()


# (generalized) linear model 
adl = linear_model.SGDClassifier(loss='squared_hinge',max_iter=1000, penalty=None)
adl.fit(X,y)

#%%
# support vector machine
gam=.2
C=10
svmod = svm.SVC(kernel='rbf', gamma=gam, C=C)
svmod.fit(X,y)


# performance
print('acc=',metrics.accuracy_score(y,adl.predict(X)))

cf = metrics.confusion_matrix(y,adl.predict(X))
print('Confusion matrix:\n%s' % cf)

# X validation
scores = model_selection.cross_val_score(svmod, X, y, cv=5)
print(scores)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

predicted = model_selection.cross_val_predict(svmod, X, y, cv=5)
print('CVacc=',metrics.accuracy_score(y, predicted))



