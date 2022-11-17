

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as matlab


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




# Show some digits


hwmny = 30
some1 = np.random.permutation(np.where(y==0)[0])[:hwmny]
some2 = np.random.permutation(np.where(y==1)[0])[:hwmny]

print(y[some2])
print(X[some2])


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



    



