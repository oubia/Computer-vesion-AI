# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:53:51 2022

@author: moham
"""
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

taska = True # here we can change it to true and it's goinng to work as it should be

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

#!/usr/bin/env python3





#%%
## Comparing algorithems base on accuracy 
###########################################
classifiers = [
    ("SGDp",1, SGDClassifier(loss='perceptron', eta0=2, learning_rate='constant', penalty=None)),
    ("Perceptron", 1,Perceptron(tol=1e-5,max_iter=100,eta0=2)),
    ("SGD",1, SGDClassifier(loss='squared_hinge', penalty=None))
]

heldout = [0.95, 0.9,0.75, 0.50, 0.25, 0.01] # Ratio of samples left out from training, for error estimation
rounds = 5 # Number of repetitions to compute average error
xx = 1. - np.array(heldout)
seed = 20
listw = []
listname = []
for name, lws, clf in classifiers:
    print("\n   Training %s" % name)
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
    
    for i in heldout:
        tr_time = 0
    
        ssumTr = 0
        ssumTs = 0
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)

            t_ini = time()
            clf.fit(X_train, y_train)
            tr_time += time() - t_ini

            y_pred = clf.predict(X_test)

            ssumTr += clf.score(X_train,y_train)
            ssumTs += clf.score(X_test,y_test)

        yyTr.append(ssumTr/rounds)
        yyTs.append(ssumTs/rounds)

        print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
        print("average accuracy: {}".format(yyTs[-1]))
    listw.append(xx)
    listw.append(yyTs)
    listname.append(name)

plt.plot(listw[0], listw[1], '-',lw=lws ,label=listname[0]+' (test)')
plt.plot(listw[2], listw[3], '-',lw=lws ,label=listname[1]+' (test)')
plt.plot(listw[4], listw[5], '-',lw=lws ,label=listname[2]+' (test)')

plt.legend(loc="lower right")
plt.xlabel("Relative training set size")
plt.ylabel("Accuracy")
plt.show()


#


#%%
## geting best Learning rate to apply it on learning
###########################################
classifiers = [    
    ("MLPclassifier",1, MLPClassifier(hidden_layer_sizes=(5), max_iter=1000, alpha=1e-4, solver="sgd",  tol=1e-4, random_state=1, learning_rate_init=.1))    
]

lerning_r = [.001,.002,.003,.004,.007,.009,.01, .02,.03, .04,.05] # Ratio of samples left out from training, for error estimation
#test all the values 
rounds = 5
heldout = 0.25
xx = 1. - np.array(heldout)
seed = np.random.randint(100)

for name, lws, clf in classifiers:
    print("\n   Training %s" % name)
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
    
    for i in lerning_r:
        tr_time = 0
        clf.set_params(learning_rate_init=i)
        ssumTr = 0
        ssumTs = 0
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=heldout, random_state=rng)

            t_ini = time()
            clf.fit(X_train, y_train)
            tr_time += time() - t_ini

            y_pred = clf.predict(X_test)

            ssumTr += clf.score(X_train,y_train)
            ssumTs += clf.score(X_test,y_test)

        yyTr.append(ssumTr/rounds)
        yyTs.append(ssumTs/rounds)

        print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
        print("average accuracy: {}".format(yyTs[-1]))
        print("learning rate is : "+str(i))    
    plt.plot( lerning_r, yyTs,'-o',lw=lws ,label=name+' (test)')


plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.xlabel("Relative learning rat")

plt.show()
#best learning rate is 0.002


#%%
# choosing the best layers
classifiers = [    
    ("MLPclassifier",1, MLPClassifier(hidden_layer_sizes=(5), max_iter=1000, alpha=1e-4, solver="sgd",  tol=1e-4, random_state=1, learning_rate_init=0.002))    
]

heldout = 0.25 # Ratio of samples left out from training, for error estimation
# here we fix the number of the neurons and we increase the number of layers
neurons = [(5),(5,5),(5,5,5),(5,5,5,5),(5,5,5,5,5),(5,5,5,5,5,5,5,5),(5,5,5,5,5,5,5,5),(5,5,5,5,5,5,5,5,5)] # Number of repetitions to compute average error
rounds = 5 #for testing the data
simple_list = [1,2,3,4,5,6,7,8]
xx = 1. - np.array(heldout)
seed = 100 #random state insialized we have to specify them not a random,graph for each seed or random state and for each graph


for name, lws, clf in classifiers:
    print("\n   Training %s" % name)
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
    
    for i,j in enumerate(neurons):
        tr_time = 0
        clf.set_params(hidden_layer_sizes=j)
        ssumTr = 0
        ssumTs = 0
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=heldout, random_state=rng)

            t_ini = time()
            clf.fit(X_train, y_train)
            tr_time += time() - t_ini

            y_pred = clf.predict(X_test)

            ssumTr += clf.score(X_train,y_train)
            ssumTs += clf.score(X_test,y_test)

        yyTr.append(ssumTr/rounds)
        yyTs.append(ssumTs/rounds)

        print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
        print("average accuracy: {}".format(yyTs[-1]))
    
    plt.plot(simple_list, yyTs, '-o',lw=lws ,label=name+' (test)')
    plt.plot(simple_list, yyTr, '--o',lw=lws, label=name+' (train)')

plt.title("Graph of number of layers")
plt.legend(loc="lower right")
plt.xlabel("Relative number of the layer")
plt.ylabel("Accuracy")
plt.show()

#%%
# number of layers is (1,1,1)
# let's see which is the best number of neurons for each layer
classifiers = [    
    ("MLPclassifier",1, MLPClassifier(hidden_layer_sizes=(5), max_iter=1000, alpha=1e-4, solver="sgd",  tol=1e-4, random_state=1, learning_rate_init=0.002))    
]

heldout = 0.25 # Ratio of samples left out from training, for error estimation
rounds = 5 #for testing the data
simple_list = range(1,41)
xx = 1. - np.array(heldout)
seed = 100 #random state insialized we have to specify them not a random,graph for each seed or random state and for each graph


for name, lws, clf in classifiers:
    print("\n   Training %s" % name)
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
    
    for i in range(1,41):
        tr_time = 0
        clf.set_params(hidden_layer_sizes=(i,i,i,i,i))

        ssumTr = 0
        ssumTs = 0
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=heldout, random_state=rng)

            t_ini = time()
            clf.fit(X_train, y_train)
            tr_time += time() - t_ini

            y_pred = clf.predict(X_test)

            ssumTr += clf.score(X_train,y_train)
            ssumTs += clf.score(X_test,y_test)

        yyTr.append(ssumTr/rounds)
        yyTs.append(ssumTs/rounds)

        print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
        print("average accuracy: {}".format(yyTs[-1]))
    
    plt.plot(simple_list, yyTs, '-o',lw=lws ,label=name+' (test)')
    plt.plot(simple_list, yyTr, '--o',lw=lws, label=name+' (train)')

plt.title("Graph of the seed number = :"+str(seed))
plt.legend(loc="lower right")
plt.xlabel("Relative number of the neurons")
plt.ylabel("Accuracy")
plt.show()

#%%plot function
from sklearn.model_selection import learning_curve

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")


    return plt
#%%
#after getting the best learning rate , layers and neurons let's see the best heldout that we can use in order to get best reseault
from sklearn import metrics

from sklearn import datasets, model_selection, linear_model, svm, cluster
from sklearn.metrics import classification_report


classifiers = [    
    ("MLPclassifier",1, MLPClassifier(hidden_layer_sizes=(4,4,4,4), max_iter=1000, alpha=1e-4, solver="sgd",  tol=1e-4, random_state=1, learning_rate_init=0.002))    
]

heldout = [0.95, 0.9,0.75, 0.50, 0.25, 0.01] # Ratio of samples left out from training, for error estimation
 # Number of repetitions to compute average error
rounds = 5 #for testing the data
xx = 1. - np.array(heldout)
simple_list = xx
seed = 100 #random state insialized we have to specify them not a random,graph for each seed or random state and for each graph

recall =[]
precision = []
for name, lws, clf in classifiers:
    print("\n   Training %s" % name)
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
    
    for i in heldout:
        tr_time = 0
    
        ssumTr = 0
        ssumTs = 0
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)

            t_ini = time()
            clf.fit(X_train, y_train)
            tr_time += time() - t_ini

            y_pred = clf.predict(X_test)

            ssumTr += clf.score(X_train,y_train)
            ssumTs += clf.score(X_test,y_test)

        yyTr.append(ssumTr/rounds)
        yyTs.append(ssumTs/rounds)

        print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
        print("average accuracy: {}".format(yyTs[-1]))
        precision.append(metrics.precision_score(y_test, y_pred)) 
        recall.append(metrics.recall_score(y_test, y_pred))
        print("Report is",classification_report(y_test,y_pred))

       
    #----
    plt.plot(heldout, precision, '-o' ,label=' (precsion)')
    plt.plot(heldout, recall, '--o',label= '(recall)')

    plt.title("Graph of the best recall and precision")
    plt.legend(loc="lower right")
    plt.xlabel("Relative number of the heldout")
    plt.ylabel("Accuracy")
    plt.show()
    #---
    
    plt.plot(simple_list, yyTs, '-o',lw=lws ,label=name+' (test)')
    plt.plot(simple_list, yyTr, '--o',lw=lws ,label=name+' (train)')

    plt.title("Graph of the best heldout")
    plt.legend(loc="lower right")
    plt.xlabel("Relative number of the heldout")
    plt.ylabel("Accuracy")
    plt.show()
    
    scores = model_selection.cross_val_score(clf, X, y, cv=6)
    plt.plot(simple_list, scores, '--o',lw=lws, label=name+' (Score)')
    plt.title("Graph of cross validation")
    plt.legend(loc="lower right")
    plt.xlabel("Relative number of the cross validation")
    plt.ylabel("Accuracy")
    plt.show()
    
    print("Coss valedation is",scores)


#let's see the crose validation
plot_learning_curve(
    MLPClassifier(hidden_layer_sizes=(4,4,4,4,4), max_iter=1000, alpha=1e-4, solver="sgd",  tol=1e-4, random_state=1, learning_rate_init=0.002),
    "MLP",
    X,
    y)
#calculate precision and recall
    
    #fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

    #plt.figure(7)
    #plt.title('ROC Curve MLP')
    #lw = 2;

    #plt.subplot(1,2,1)
    #plt.plot(fpr, tpr, color="darkorange", lw = lw, label="ROC curve")
    #plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel("False Positive Rate")
    #plt.ylabel("True Positive Rate")
    #plt.title("MLP: ROC Curve WITH PCA")
    #plt.legend(loc="lower right")
    #plt.show()

#95 is the best 
#%%
#SVM linear model here we tried to get the best parameters in order to get a highier accuracy
from sklearn.svm import SVC
from sklearn import datasets, model_selection, linear_model, svm, cluster
import seaborn as sns
from sklearn import metrics, decomposition
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# support vector machine

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train) 

ax = plt.gca()
xlim = ax.get_xlim()
C = .00001  # SVM regularization parameter
gam= 'scale'
classifier = svm.SVC(kernel='linear',gamma=gam,C=C)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y_test.shape)
print(classification_report(y_test,y_pred))

#ax.scatter(X_test[:,0],X_test[:,1],c=y_pred,cmap ='winter',marker='s')

'''w = classifier.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(xlim[0],xlim[1])
yy = a * xx - (classifier.intercept_[0]/w[1])
plt.plot(xx,yy)
plt.show()'''

#this function help use to get the best parameters C and gamma
params = [
    {'C':[.0000001,.00001,.0001,.001,.01,.1,0.5,1,10,100],
     'gamma':['scale',1,.1,.01,.001,.0001,.00001],
     'kernel':['linear'],
     }
    
    ]
opt_params = GridSearchCV(
    SVC(),
    params,
    cv=8,
    scoring='accuracy',
    verbose=0
    )
opt_params.fit(X_train,y_train)
print(opt_params.best_params_)
plot_learning_curve(
    svm.SVC(kernel='linear',gamma=gam,C=C),
    "SVM Linear",
    X,
    y)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train) 

ax = plt.gca()
xlim = ax.get_xlim()
C = .5  # SVM regularization parameter
gam= 'scale'
svmod = svm.SVC(kernel='sigmoid',gamma=gam,C=C)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y_test.shape)
print(classification_report(y_test,y_pred))

#ax.scatter(X_test[:,0],X_test[:,1],c=y_pred,cmap ='winter',marker='s')
'''plt.scatter(
        classifier.support_vectors_[:, 0],
        classifier.support_vectors_[:, 1],
        s=80,
        facecolors="none",
        zorder=10,
        edgecolors="k",
    )
#plt.show()'''


params = [
    {'C':[0.5,1,10,100],
     'gamma':['scale',1,.1,.01,.001,000.1],
     'kernel':['sigmoid'],
     }
    
    ]
opt_params = GridSearchCV(
    SVC(),
    params,
    cv=5,
    scoring='accuracy',
    verbose=0
    )
opt_params.fit(X_train,y_train)
print(opt_params.best_params_)
plot_learning_curve(
    svm.SVC(kernel='sigmoid',gamma=gam,C=C),
    "SVM sigmoid",
    X,
    y)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

ax = plt.gca()
xlim = ax.get_xlim()
C = 10  # SVM regularization parameter
gam= 'scale'
classifier = svm.SVC(kernel='rbf',gamma=gam,C=C)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y_test.shape)
print(classification_report(y_test,y_pred))
'''ax.scatter(X_test[:,0],X_test[:,1],c=y_pred,cmap ='winter',marker='s')
plt.scatter(
        classifier.support_vectors_[:, 0],
        classifier.support_vectors_[:, 1],
        s=80,
        facecolors="none",
        zorder=10,
        edgecolors="k",
    )
plt.show()


plt.show()'''

#to get the best parameters
params = [
    {'C':[.0001,.001,.01,.1,0.5,1,10,100],
     'gamma':['scale',1,.1,.01,.001,000.1],
     'kernel':['rbf'],
     }
    
    ]
opt_params = GridSearchCV(
    SVC(),
    params,
    cv=5,
    scoring='accuracy',
    verbose=0
    )
opt_params.fit(X_train,y_train)
print(opt_params.best_params_)
plot_learning_curve(
    svm.SVC(kernel='rbf',gamma=gam,C=C),
    "SVM rbf",
    X,
    y)

#%%
#%%
#--------------------- k-NN classifiers ---------------------------

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.25)

R_list_accuraacy = []
wknn = KNeighborsClassifier()
for k in range (1,40):
    wknn.set_params(n_neighbors=k)
    my_model = wknn.fit(X_train,Y_train)
    predected_value = my_model.predict(X_test)
    auc = metrics.roc_auc_score(Y_test,  predected_value)
    R_list_accuraacy.append (auc)



#---------------cross validation ----------------------------------------------
from sklearn.model_selection import cross_val_score

errorRate = []
wknn = KNeighborsClassifier()
for k in range (1,40):
    wknn.set_params(n_neighbors=k)
    error_rates = cross_val_score(wknn, X, y, cv=2)
    errorRate.append((error_rates[0]+error_rates[1])/2)
    

print(len(errorRate))
print(len(R_list_accuraacy))


plt.plot(range(1,40),R_list_accuraacy,color='blue',label="Auccoracy rate")
plt.plot(range(1,40),errorRate,color='red',label="cross validation")
plt.title("graph with a cross validation")
plt.ylabel('Accuracy')
plt.xlabel('K-nn')
plt.legend(loc=4)
plt.show()
#%%

#--------------------- k-NN classifiers  + grand nombre a ACCURACY = 4 ---------------------------


from sklearn.model_selection import cross_val_score

X_train,X_test,Y_train,Y_test = X,X,y,y

R_list_accuraacy = []
wknn = KNeighborsClassifier()
for k in range (1,4):
    wknn.set_params(n_neighbors=k)
    my_model = wknn.fit(X_train,Y_train)
    predected_value = my_model.predict(X_test)
    auc = metrics.roc_auc_score(Y_test,  predected_value)
    R_list_accuraacy.append (auc)

#---------------cross validation ----------------------------------------------
from sklearn.model_selection import cross_val_score

errorRate = []
wknn = KNeighborsClassifier()
for k in range (1,4):
    wknn.set_params(n_neighbors=k)
    error_rates = cross_val_score(wknn, X, y, cv=2)
    errorRate.append((error_rates[0]+error_rates[1])/2)
    

print(len(errorRate))
print(len(R_list_accuraacy))


plt.plot(range(1,4),R_list_accuraacy,color='blue',label="Auccoracy rate")
plt.plot(range(1,4),errorRate,color='red',label="cross validation")
plt.title("graph with a cross validation")
plt.ylabel('Accuracy')
plt.xlabel('K-nn')
plt.legend(loc=4)
plt.show()
#%%
#-------------------------------KNN HALDOUT-----------------------------------

print(y)
print(X)

hold = [.25,.50,.75,.90]

X0 = X
Y0 = y

for i in hold:
    X0_train,X0_test,Y0_train,Y0_test = train_test_split(X0,Y0, test_size=i)

    #Holdout validation for X0------------------------------
    wc0= KNeighborsClassifier(n_neighbors=4)
    my_model0 = wc0.fit(X0_train,Y0_train)
    
    #testing the model
    predicted_value0 =my_model0.predict(X0_test)
    
    matrix_x0 = metrics.confusion_matrix(Y0_test,predicted_value0)
    
    #Roc curve
    fpr, tpr, _ = metrics.roc_curve(Y0_test,   predicted_value0)
    auc = metrics.roc_auc_score(Y0_test,  predicted_value0)
    plt.plot(fpr,tpr,label="AUC="+str(auc)+"holdout = "+str(i))
plt.title("graph of Knn  with a holdout validation number= ")
plt.ylabel('Accuracy')
plt.xlabel('Holdout')
plt.legend(loc=4)
plt.show()



#%%




#%%


#%%
from sklearn import metrics, decomposition

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
classes=range(2)
plt.scatter(Xred[:,0],Xred[:,1],c=y,s=6)
for i in range(2):
    plt.scatter([],[],label=i)
    
plt.plot(Cred[:,0],Cred[:,1],'d',markersize=10,markeredgecolor='k',markerfacecolor='red',label='centers')
    
plt.xlim(tuple(map(sum,zip((0,10),plt.xlim()))))
plt.legend(loc='upper right')
plt.show()
