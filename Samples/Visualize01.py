# -*- coding: utf-8 -*-
"""
 * Author:    Andrea Casalino
 * Created:   26.12.2019
*
* report any bug to andrecasa91@gmail.com.
"""

import matplotlib.pyplot as plt
import numpy
from random import random

def get_cols(mat, c):
    C = []
    for r in mat:
        C.append(r[c])
    return C

def get_pos(labels):
    n_labels = int(max(labels)) + 1
    pos = []
    for k in range(0, n_labels):
        pos.append([])
    for p in range(0, len(labels)):
        pp = int(labels[p])
        pos[pp].append(int(p))
    return pos


def plot_K_means():
    mat = numpy.loadtxt('K_means_clustering')
    clusters_pos = get_pos(get_cols(mat,0))
    colors = []
    names = []
    for i in range(0, len(clusters_pos)):
        colors.append([random(), random(), random()])
        names.append('cluster' + str(i+1))
    for i in range(0, len(clusters_pos)):
        temp = mat[clusters_pos[i]]
        plt.plot(get_cols(temp,1), get_cols(temp,2), '*', color=colors[i])
    plt.legend(names)
    

plt.figure()
plot_K_means()
plt.axis('equal')   
plt.title('Results of K-means clustering') 
plt.show() 