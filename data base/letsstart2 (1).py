

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as matlab
import numpy as np
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


# Load the database
mat_file = "E:/homy/S6/Machine learning/Final project/data base/myLFW.mat"
mat = matlab.loadmat(mat_file, squeeze_me=True)  # dictionary
list(mat.keys())  # list vars

print(mat)
taska = True

data = mat["data"]      # read feature vectors
labs = mat["nlab"]      # read labels 1..10

allNlabs = np.unique(labs)  # all labs 0 .. 9

classsiz = ()
for c in allNlabs:
    classsiz = classsiz + (np.size(np.nonzero(labs == c)),)
print('\n%% Class labels are: %s' % (allNlabs,))
print('%% Class frequencies are: %s' % (classsiz,))


# -----------------------------generatewildAll premier parag--------------------
MinClassSize = 53    # with this value we retrieve c=10 classes

FinalClassSize = 50  # then we select an equal size for all <= MinClassSize

FaceScale = 0.4   # we reduce the resolution and hence the final feature space dimension

lfw_people = fetch_lfw_people(
    min_faces_per_person=MinClassSize, resize=FaceScale, funneled=True)
n_samples, h, w = lfw_people.images.shape

# ------------------------generatewildAll premier parag-------
# target_names = lfw_people.target_names # strings
target_names = np.array(["myFace", "otherFaces"])  # strings
n_classes = target_names.shape[0]


# Let's say my face is ...

myFace = 3

otherFaces = np.setdiff1d(allNlabs, myFace)
other3Faces = np.random.permutation(otherFaces)[:3]

if taska:
    others = otherFaces
else:
    others = other3Faces

print('class 1 = %s' % myFace)
print('class 2 = %s' % others)

# To construct a 2-class dataset you can use the same matrix
# data and change the vector of labels

aux = labs
classone = np.in1d(labs, myFace)
classtwo = np.in1d(labs, others)
aux[classone] = 0  # class one
aux[classtwo] = 1  # class two

# Features
X = data[np.logical_or(classone, classtwo)]
# (unchanged) labels
y = aux[np.logical_or(classone, classtwo)]


# Show some digits


hwmny = 30

some1 = np.random.permutation(np.where(y == 0)[0])[:hwmny]
some2 = np.random.permutation(np.where(y == 1)[0])[:hwmny]


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

# shows first 30 images
#plot_gallery(newData, newLab, h, w, n_row=5, n_col=6)


# shows examples of the two first classes
w = 37
h = 50


plot_gallery(X[some1, :], y[some1], h, w, n_row=4, n_col=6)
plot_gallery(X[some2, :], y[some2], h, w, n_row=4, n_col=6)


# --------------------------------------------copie generateWildAll deuxiéme parag -----------------------------------
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


###############################################################################
# PCA computation  Principal Component Analysis

def pca_1(xtrain, xtest, c):
    # whiten : bool, optional argument and by default "False."
    # Whitening removes some information from the transformed signal but improves the predictive accuracy

    start = time()
    n_components = 100
    print("Computing PCA.....")
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(xtrain)  # standardizing the training data to mean =0 & variance =1

    # # apply PCA transformation to training data
    X_train_pca = pca.transform(xtrain)
    X_test_pca = pca.transform(xtest)
    end = time()
    print("Time taken to compute PCA: %f sec" % ((end-start)))


###############################################################################
# MLP classification

# train a multi-layer perceptron
# verbose : bool, optional, default False (Whether to print progress messages to stdout)
# batch_size :  number of samples that will be propagated through the network

start = time()
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256,
                    verbose=False, early_stopping=True).fit(X_train, y_train)
end = time()
print("Time taken to train MLP: %f sec" % (end-start))
start = time()
y_pred = clf.predict(X_test)
end = time()
print("Time taken by MLP to predict: %f sec" % (end-start))
print("classfication report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


fpr, tpr, _ = metrics.roc_curve(y_test.ravel(),clf.predict_proba(X_test)[:,1])
auc = metrics.accuracy_score(y_test,  y_pred)

plt.figure()

plt.plot(fpr,tpr,label="AUC="+str(auc))

plt.title("graph of Roc curve")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

# def titles(y_pred, y_test, target_names):
#     for i in range(y_pred.shape[0]):
#         pred_name = target_names[y_pred[i]].split(' ')[-1]
#         true_name = target_names[y_test[i]].split(' ')[-1]
#         yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

###############################################################################
# Visualization of the results
# def plot_images(titles, images, height, width, rows, columns):
#     plt.figure(figsize=(columns * 1.5,rows * 2.5))
#     #adjust the image dimensions by tweaking these parameters
#     plt.subplots_adjust(bottom=.90, left=.05, right=0.90, top=.02, hspace=.40)
#     for i in range(columns * rows):
#         plt.subplot(rows, columns, i + 1)
#         plt.imshow(images[i].reshape((height, width)), cmap=plt.cm.gray)
#         plt.title(titles[i], size=10)
#         plt.xticks(())
#         plt.yticks(())


# -----------------------------------------------------------------------------------
# Código de ejemplo, para sacar más información.

# Clustering
kmn = cluster.KMeans(init='k-means++', n_clusters=2, n_init=2)
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
classes = range(2)
plt.scatter(Xred[:, 0], Xred[:, 1], c=y, s=6)
for i in range(2):
    plt.scatter([], [], label=i)

plt.plot(Cred[:, 0], Cred[:, 1], 'd', markersize=10,
         markeredgecolor='k', markerfacecolor='w', label='centers')

plt.xlim(tuple(map(sum, zip((0, 2), plt.xlim()))))
plt.legend(loc='upper right')
plt.show()

# (generalized) linear model
adl = linear_model.SGDClassifier(
    loss='squared_hinge', max_iter=1000, penalty=None)
adl.fit(X, y)

# support vector machine
gam = .2
C = 10
svmod = svm.SVC(kernel='rbf', gamma=gam, C=C)
svmod.fit(X, y)
