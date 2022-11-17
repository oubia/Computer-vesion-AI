# -*- coding: utf-8 -*-
"""
Created on Sat May 14 20:37:42 2022

@author: moham
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as matlab
from sklearn.datasets import fetch_lfw_people
from scipy.io import savemat


from time import time
from sklearn.decomposition import PCA 
from sklearn import datasets, model_selection, linear_model, svm, cluster
from sklearn import metrics, decomposition
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


# Load the database
mat_file =  "E:/homy/S6/Machine learning/Final project/data base/myLFW.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary
list(mat.keys()) # list vars

data = mat["data"]      # read feature vectors
labs = mat["nlab"]      # read labels 1..10

allNlabs = np.unique(labs) # all labs 0 .. 9

classsiz = ()
for c in allNlabs:
    classsiz = classsiz + (np.size(np.nonzero(labs==c)),)  
print ('\n%% Class labels are: %s' % (allNlabs,))
print ('%% Class frequencies are: %s' % (classsiz,))


MinClassSize = 53    # with this value we retrieve c=10 classes

FinalClassSize = 50  # then we select an equal size for all <= MinClassSize

FaceScale = 0.4   # we reduce the resolution and hence the final feature space dimension

lfw_people = fetch_lfw_people(min_faces_per_person=MinClassSize, resize=FaceScale, funneled=True)
n_samples, h, w = lfw_people.images.shape

target_names = np.array(["Schroeder","others"])
n_classes = target_names.shape[0]

# Let's say my face is ...

myFace = 4

otherFaces  = np.setdiff1d(allNlabs,myFace)
# other3Faces = np.random.permutation(otherFaces)[1:4]
other3Faces = otherFaces[0:3]

taska = False

if taska:
    others = otherFaces
else:
    others = other3Faces

print('class 1 = %s' % myFace)
print('class 2 = %s' % others)

# To construct a 2-class dataset you can use the same matrix
# data and change the vector of labels

aux = labs
classone = np.in1d(labs,myFace)
classtwo = np.in1d(labs,others)
aux[classone] = 0  # class one   ## OUR FACE
aux[classtwo] = 1  # class two   ## 3 OTHER FACES

# Features
X = data[np.logical_or(classone,classtwo)]
# (unchanged) labels
y = aux[np.logical_or(classone,classtwo)]


# Show some digits
hwmny = 50

some1 = np.random.permutation(np.where(y==0)[0])[:hwmny]
some2 = np.random.permutation(np.where(y==1)[0])[:hwmny]



def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
        

# shows first 30 images
#plot_gallery(newData, newLab, h, w, n_row=5, n_col=6)  
        
# shows examples of the two first classes   
w = 37     
h = 50
    
    
plot_gallery(X[some1,:], y[some1], h, w, n_row=5, n_col=10)

print(y)
#%%
# plot_gallery(X[some2,:], y[some2], h, w, n_row=5, n_col=10)

###############################################################################
# PCA computation 

# def pca_1 (xtrain, xtest, c):
#       #whiten : bool, optional argument and by default "False."
#       #Whitening removes some information from the transformed signal but improves the predictive accuracy
 
#       start =time()
#       n_components =100;
#       print("Computing PCA.....")
#       pca = PCA(n_components=n_components, whiten=True)
#       pca.fit(xtrain)      #standardizing the training data to mean =0 & variance =1 
    
#     # # apply PCA transformation to training data
#       X_train_pca = pca.transform(xtrain)
#       X_test_pca = pca.transform(xtest)
#       end = time()
#       print("Time taken to compute PCA: %f sec" % ((end-start)))
     
###############################################################################
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5)
# print('y',y)
# print('ytest',y_test)
# print('ytrain',y_train)

##############################################################################
# print('X_train ' , X_train)
# print('X_test ', X_test)
# print('y_train ' , y_train)
# print('y_test ' , y_test)
 
###############################################################################
#MLP classification
 
 # train a multi-layer perceptron
 #verbose : bool, optional, default False (Whether to print progress messages to stdout)
 #batch_size :  number of samples that will be propagated through the network
 
 #PCA

n_components = 50
pca_mlp = RandomizedPCA(n_components = n_components, whiten=True).fit(X_train)

eigenfaces_mlp = pca_mlp.components_.reshape((n_components, h, w))

X_train_pca = pca_mlp.transform(X_train)
X_test_pca = pca_mlp.transform(X_test)
 
# MLP WITH PCA
print("########### MLP WITH PCA #############")
start =time()
print("Fitting the classifier to the training set")
clf1 = MLPClassifier(hidden_layer_sizes=(100,75), verbose=False, max_iter=1000).fit(X_train_pca, y_train)
end = time()
print("Time taken to train MLP: %f sec" % (end-start))
start = time()
y_pred = clf1.predict(X_test_pca)
end = time()
print("Time taken by MLP to predict: %f sec" % (end-start))
print("classfication report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
# print('ypred',y_pred)
# print('ypred',y_test)     


plt.figure(5)
plt.title('Loss Curve MLP')
# fig, axes = plt.subplots(1, 2, figsize=(30,10))
plt.subplot(1,2,1)
plt.title('MLP: WITH PCA')
plt.plot(clf1.loss_curve_)

#The most popular example of a learning curve is loss over time. 
#Loss (or cost) measures our model error, or “how bad our model is doing”. 
#So, for now, the lower our loss becomes, the better our model performance will be.

# MLP WITHOUT PCA
print("########### MLP WITHOUT PCA #############")
start =time()
print("Fitting the classifier to the training set")
clf2 = MLPClassifier(hidden_layer_sizes=(100,75), verbose=False, max_iter=1000).fit(X_train, y_train)
end = time()
print("Time taken to train MLP: %f sec" % (end-start))
start = time()
y_pred = clf2.predict(X_test)
end = time()
print("Time taken by MLP to predict: %f sec" % (end-start))
print("classfication report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
# print('ypred',y_pred)
# print('ypred',y_test)     
# print('clf loss', clf.loss_curve_)
plt.subplot(1,2,2)
plt.plot(clf2.loss_curve_)
plt.title('MLP: WITHOUT PCA')
plt.show()



fpr2_1, tpr2_1, threshold1 = roc_curve(y_test, clf1.predict_proba(X_test_pca)[:,1])
fpr2_2, tpr2_2, threshold1 = roc_curve(y_test, clf2.predict_proba(X_test)[:,1])

plt.figure(7)
plt.title('ROC Curve MLP')
lw = 2;

plt.subplot(1,2,1)
plt.plot(fpr2_1, tpr2_1, color="darkorange", lw = lw, label="ROC curve")
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("MLP: ROC Curve WITH PCA")
plt.legend(loc="lower right")
plt.show()

plt.subplot(1,2,2)
plt.plot(fpr2_2, tpr2_2, color="darkorange", lw = lw, label="ROC curve")
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("MLP: ROC Curve WITHOUT PCA")
plt.legend(loc="lower right")
plt.show()


###############################################################################
# CLUSTERING CLASSIFIER K-Means
#%%
# Clustering 
kmn = cluster.KMeans(init='k-means++', n_clusters=2, n_init=2)
kmn.fit(X)
centers = kmn.cluster_centers_
labels = kmn.labels_
score = kmn.inertia_



# Dimensionality reduction using PCA
pca = decomposition.PCA(n_components=None)
pca.fit(X)
pca.fit
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
classes=range(2)
plt.scatter(Xred[:,0],Xred[:,1],c=y,s=6)
for i in range(2):
    plt.scatter([],[],label=i)
    
plt.plot(Cred[:,0],Cred[:,1],'d',markersize=10,markeredgecolor='k',markerfacecolor='w',label='centers')
    
plt.xlim(tuple(map(sum,zip((0,10),plt.xlim()))))
plt.legend(loc='upper right')
plt.show()