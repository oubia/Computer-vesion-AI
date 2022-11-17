#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Ignacio García Fernández

Load relief database 
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.io import loadmat

def draw_terrain(Z,N=None,label="No label",d=1.7):    

    if N is None:
        tam = len(Z)
    else:
        tam = 2**N+1
    
    A = np.max(abs(Z))
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(-d*A,d*A)

    
    X,Y = np.meshgrid( range(tam),range(tam) )

    #ax.plot_wireframe(X, Y, Z)
    ax.plot_surface(X, Y, Z)
    if np.max(np.abs(Z)) > 1e-5:
        ax.contour(X,Y,Z,dir='z',offset=-d*A)
    
    plt.title(label)
    plt.show()

#

def get_patch(X,i,dim=8):
    ptch = X[i].reshape(dim,dim)
    return ptch

def show_patch(X,i,dim=8):
    """ Auxiliary function to show a digit """
    plt.gray()
    plt.matshow(get_patch(X,i,dim))
    plt.title("A sample digit: "+str(y[i]))
    plt.show()
#


#%% Load a dataset and plot some samples
d = 65

mat = loadmat("terrain.mat", squeeze_me=True, struct_as_record=False)
X = np.array(mat['A'])/4.0 + 0.5
y = mat['y']



examp = [1,int(len(X)/4)+1,int(len(X)/2)+1,3*int(len(X)/4)+1]

for ex in examp:
    show_patch(X,ex,dim=d)
    draw_terrain(get_patch(X,ex,dim=d),d=1)


# Plot images of the height fields
n_img_per_row = 10
n_rows = 10
h = d+2 # height/width of each digit 
img = np.zeros(( h * n_rows , h * n_img_per_row ))
for i in range(n_rows):
    ix = h * i + 1
    for j in range(n_img_per_row):
        iy = h * j + 1
        k = np.random.randint(len(X))
        img[ix:ix + d, iy:iy + d] = get_patch(X,k,dim=d)

plt.imshow(img, cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the mountains relief dataset')
plt.show()

