# -*- coding: utf-8 -*-
"""
 * Author:    Andrea Casalino
 * Created:   26.12.2019
*
* report any bug to andrecasa91@gmail.com.
"""
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import json
import numpy as np


def get_json_from_file(name):
    with open(name) as json_file:
        return json.load(json_file)

def get_R(covariance_matrix):
    m = np.matrix(covariance_matrix)
    E, R = np.linalg.eig(m)
    for i in range(0, len(E)) :
        R[:,i] = R[:,i] * np.sqrt(E[i]) 
    return R
       
def plot_Polygon(V, ax, col, alp):
    patches = []
    polygon = Polygon(V, True)
    patches.append(polygon)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=alp, color=col)
    ax.add_collection(p)
 
def plot_cluster(ax, cluster, color):
       
    teta = np.linspace(0, 2 * np.pi, num=50)
    C = []
    R = get_R(cluster['Covariance']);

    for t in teta:
        new_x = R[0,0] * np.cos(t) + R[0,1] * np.sin(t) +  cluster['Mean'][0]
        new_y = R[1,0] * np.cos(t) + R[1,1] * np.sin(t) +  cluster['Mean'][1]
        C.append([new_x, new_y])

    plot_Polygon(C , ax, color, cluster['w'])
    ax.plot([cluster['Mean'][0]] ,[cluster['Mean'][1]] , '*k')
    ax.text(cluster['Mean'][0] ,cluster['Mean'][1], r"$w$=" + str(cluster['w']), color='black')


def plot_GMM(file, ax, color):
    data = get_json_from_file(file)
    
    for k in range(0, len(data)):
        plot_cluster(ax, data[k], color)
    ax.plot(0, 0 , '.', markersize=0.001)
    
fig, ax = plt.subplots()
lim = plot_GMM('random_model.json',ax,[0,1,0])
plt.title('real model cluster covariances (trasparency proportional to the clusters weigths)')
plt.show() 

fig, ax = plt.subplots()
lim = plot_GMM('learnt_model.json',ax,[0,1,0])
plt.title('learnt model cluster covariances (trasparency proportional to the clusters weigths)')
plt.show() 