#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:51:42 2019

@author: pfr

Generates a balanced datset corresponding to the 10 most populated classes 
in the LFW dataset (see scikit-learn documentation)

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from scipy.io import savemat


#%% #############################################################################

MinClassSize = 53    # with this value we retrieve c=10 classes

FinalClassSize = 50  # then we select an equal size for all <= MinClassSize

FaceScale = 0.4      # we reduce the resolution and hence the final feature space dimension

lfw_people = fetch_lfw_people(min_faces_per_person=MinClassSize, resize=FaceScale, funneled=True)
n_samples, h, w = lfw_people.images.shape

# input feature vectors
X = lfw_people.data
n_features = X.shape[1]

# ground truth labels
y = lfw_people.target                  # numeric
target_names = lfw_people.target_names # strings
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("img_size: ")
print(lfw_people.images[0].shape)
print("n_classes: %d" % n_classes)

HowMany = FinalClassSize
newData = np.zeros([HowMany*n_classes,n_features])
newLab  = np.zeros([HowMany*n_classes,1],dtype=np.int16)

for p in range(n_classes):
    #print(X[y==p][:HowMany,:].shape)
    newData[ p*HowMany:(p+1)*HowMany , :] = np.copy(X[y==p][:HowMany,:]) 
    newLab[ p*HowMany:(p+1)*HowMany ] = p

print("\nFinal dataset size:")
print("n_samples: %d" % newLab.size)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print("n_samples_per_class: %d" % FinalClassSize)


# save the balanced dataset to use it in Matlab
#savemat("myLFW", dict(data=newData, nlab=newLab,
#                      name="LFW", featsize=[h,w], lablist=lfw_people.target_names ) )

# load and use it as:
# >> load('myLFW.mat');
# >> A=dataset(data,nlab,'featsize',featsize,'lablist',lablist,'name',name)
# >> clear data featsize lablist name nlab


# #############################################################################
# Plot some vectors as images

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
howm=15        
plot_gallery(newData[FinalClassSize-howm:FinalClassSize+howm,:], newLab[FinalClassSize-howm:FinalClassSize+howm], h, w, n_row=4, n_col=6)
