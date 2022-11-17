# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:54:10 2022

@author: hpp
"""

from sklearn import datasets,metrics
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice
import scipy.io.matlab as matlab


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis



#%%
# Load the database
mat_file =  "C:/Users/hpp/Desktop/code/myLFW.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary
list(mat.keys()) # list vars

print(mat)
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

otherFaces  = np.setdiff1d(allNlabs,myFace)
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
classone = np.in1d(labs,myFace)
classtwo = np.in1d(labs,others)
aux[classone] = 0  # class one
aux[classtwo] = 1  # class two

# Features
X = data[np.logical_or(classone,classtwo)]
# (unchanged) labels
y = aux[np.logical_or(classone,classtwo)]


#%%
#---------------------------------Holdout validation------------------------------------------
#----------------------------------Algo : QuadraticDiscriminantAnalysis -----------------------

print(y)
print(X)

hold = [.25,.50,.75,.90]

X0 = X
Y0 = y

for i in hold:
    X0_train,X0_test,Y0_train,Y0_test = train_test_split(X0,Y0, test_size=i)

    #Holdout validation for X0------------------------------
    wc0= QuadraticDiscriminantAnalysis()
    my_model0 = wc0.fit(X0_train,Y0_train)
    
    #testing the model
    predicted_value0 =my_model0.predict(X0_test)
    
    matrix_x0 = metrics.confusion_matrix(Y0_test,predicted_value0)
    
    #Roc curve
    fpr, tpr, _ = metrics.roc_curve(Y0_test,  wc0.decision_function(X0_test))
    auc = metrics.roc_auc_score(Y0_test,  predicted_value0)
    plt.plot(fpr,tpr,label="AUC="+str(auc)+"holdout = "+str(i))
plt.title("graph of X0  with a holdout validation number= ")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


#%%




#---------------------------------resubstitution validation------------------------------------------
#----------------------------------Algo : QuadraticDiscriminantAnalysis -----------------------

X0 = X
Y0 = y

X0_train,X0_test,Y0_train,Y0_test = X0,X0,Y0,Y0



# validation for X0------------------------------
wc0= QuadraticDiscriminantAnalysis()
my_model0 = wc0.fit(X0_train,Y0_train)

#testing the model
predicted_value0 =my_model0.predict(X0_test)
matrix_x0 = metrics.confusion_matrix(Y0_test,predicted_value0)

#Roc curve
fpr, tpr, _ = metrics.roc_curve(Y0_test,  wc0.decision_function(X0_test))
auc = metrics.roc_auc_score(Y0_test,  predicted_value0)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.title("graph of X0 with a resubstitution validation")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#%%

#-------------------------------------resubstitution validation-------------------------
#-----------------------------------Algo : Linear classifiers---------------------------------------

X0 = X
Y0 = y

X0_train,X0_test,Y0_train,Y0_test = X0,X0,Y0,Y0


wcQ0 = LinearDiscriminantAnalysis()

my_model0 = wcQ0.fit(X0_train,Y0_train)

#testing the model
predicted_value0 =my_model0.predict(X0_test)
matrix_x0 = metrics.confusion_matrix(Y0_test,predicted_value0)

#Roc curve
fpr, tpr, _ = metrics.roc_curve(Y0_test,  wcQ0.decision_function(X0_test))
auc = metrics.roc_auc_score(Y0_test,  predicted_value0)
plt.plot(fpr,tpr,label="AUC="+str(auc),color='red')
plt.title("graph of X0  with a resubstitution validation")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
#%%
#-------------------------------------Holdout validation-------------------------
#-----------------------------------Algo : Linear classifiers---------------------------------------


hold = [.25,.50,.75,.90]

X0 = X
Y0 = y

for i in hold:
    X0_train,X0_test,Y0_train,Y0_test = train_test_split(X0,Y0, test_size=i)
    
    #Holdout validation for X0------------------------------
    wc0= LinearDiscriminantAnalysis()
    my_model0 = wc0.fit(X0_train,Y0_train)
    
    #testing the model
    predicted_value0 =my_model0.predict(X0_test)
    
    matrix_x0 = metrics.confusion_matrix(Y0_test,predicted_value0)
    
    #Roc curve
    fpr, tpr, _ = metrics.roc_curve(Y0_test,  wc0.decision_function(X0_test))
    auc = metrics.roc_auc_score(Y0_test,  predicted_value0)
    plt.plot(fpr,tpr,label="AUC="+str(auc)+"holdout = "+str(i))
plt.title("graph X0 with a holdout validation="+str(i))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()



#%%



#%%




