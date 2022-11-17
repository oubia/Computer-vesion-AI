"""
==================================================
Code mostly taken from scikit-learn

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
import scipy.io.matlab as matlab
from sklearn.metrics import roc_curve, auc



# Load the database
mat_file =  "E:/homy/S6/Machine learning/Final project/data base/myLFW.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary
list(mat.keys()) # list vars

taska = True

data = mat["data"]      # read feature vectors
labs = mat["nlab"]      # read labels 1..10

allNlabs = np.unique(labs) # all labs 0 .. 9

classsiz = ()
for c in allNlabs:
    classsiz = classsiz + (np.size(np.nonzero(labs==c)),)  
print ('\n%% Class labels are: %s' % (allNlabs,))
print ('%% Class frequencies are: %s' % (classsiz,))


# Let's say my face is ...

myFace = 3

otherFaces  = np.setdiff1d(allNlabs,myFace)#b
other3Faces = np.random.permutation(otherFaces)[:3]#a

if taska:
    others = otherFaces,
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




# Show some digits


hwmny = 30
some1 = np.random.permutation(np.where(y==0)[0])[:hwmny]
some2 = np.random.permutation(np.where(y==1)[0])[:hwmny]

print(y)
print(X)



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

#X[some1],y[some1] //tregeted

#X[some2],y[some2] //data

plot_gallery(X[some1,:], y[some1], h, w, n_row=4, n_col=6)
plot_gallery(X[some2,:], y[some2], h, w, n_row=4, n_col=6)
#%%
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
# select 2 dimensions and 2 classes
#whichD = [0,2];Classes = [1,2]
#whichD = [1,2];Classes = [1,2]
whichD = [0,1];Classes = [0,1]   # linearly separable

whichC = y

#X,y = datasets.make_moons(100,noise=.1)






eps = np.spacing(np.float32(1.0))

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1  # SVM regularization parameter
deg = 5
gam= .01
models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='rbf', gamma=gam, C=C),
          svm.SVC(kernel='poly', degree=deg, C=C))

models = (clf.fit(X, y) for clf in models)


# title for the plots
titles = ('linear, C={}'.format(C),
        'rbf, C={}, g={}'.format(C,gam),
        'poly, C={}, d={}'.format(C,deg))

# Set-up plotting.
fig=plt.figure(1)
fig.clf()
fig, sub = plt.subplots(1, len(titles) ,num=1,figsize=(10,4))

plt.subplots_adjust(wspace=.4, hspace=0.4)


X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)


for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.6)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=22, edgecolors=None)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    svs = clf.support_
    ax.scatter(X0[svs], X1[svs], c=y[svs], cmap=plt.cm.coolwarm, s=22, edgecolors='w')
    # give some info
    #print(clf.dual_coef_)  # Alphas
    print(title,' Acc=',clf.score(X,y),', Nsv=',clf.n_support_,', NsvIN=',  np.sum(np.abs(clf.dual_coef_)>=C-eps)            )

plt.show()
