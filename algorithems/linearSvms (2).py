"""
==================================================
Code mostly taken from scikit-learn

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


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
import seaborn as sns

# Load the database
mat_file =  "E:/homy/S6/Machine learning/Final project/data base/myLFW.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary
list(mat.keys()) # list vars

taska = False

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

target_names = np.array(["my face","other faces"])
n_classes = target_names.shape[0]

# Let's say my face is ...

myFace = 4

otherFaces  = np.setdiff1d(allNlabs,myFace)
other3Faces = np.random.permutation(otherFaces)[1:4]

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
aux[classone] = 0  # class one
aux[classtwo] = 1  # class two



# Features
X = data[np.logical_or(classone,classtwo)]
# (unchanged) labels
y = aux[np.logical_or(classone,classtwo)]


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the margin of an SVM classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    D = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = D>1
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z,colors=('w','w','b','b'),alpha=.6) # **params)
    Z = D<-1
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z,colors=('w','w','r','r'),alpha=.6) #**params)
    
    out = ax.contour(xx,yy,D.reshape(xx.shape),linewidths=[.1,1,.1], linestyles=['-','--','-'],levels=[-1,0,1], colors=['r','k','b'])
    
    return out


##########################################################################
# import data
iris = datasets.load_iris()
# select 2 dimensions and 2 classes
#whichD = [0,2];Classes = [1,2]
#whichD = [1,2];Classes = [1,2]
whichD = [0,1];Classes = [0,1]   # linearly separable

# fnames = [iris.feature_names[i] for i in Classes]
# whichC = np.in1d(iris.target,Classes)
# X = iris.data[whichC,:]
# X = X[:,whichD]
# y = iris.target[whichC]

#X,y = datasets.make_moons(100,noise=.1)






eps = np.spacing(np.float32(1.0))

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors

