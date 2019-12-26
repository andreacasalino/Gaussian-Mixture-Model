# -*- coding: utf-8 -*-
"""
 * Author:    Andrea Casalino
 * Created:   26.12.2019
*
* report any bug to andrecasa91@gmail.com.
"""
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
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
    

def plot_cluster(ax, cluster, color):
    angles = np.linspace(0, 2 * np.pi, num=100)
    C = [[],[]]
    for a in angles:
        C[0].append(2.5 * np.cos(a))
        C[1].append(2.5 * np.sin(a))
    
    R = get_R(cluster['Covariance']);
    
    x_lim=[0,0]
    y_lim=[0,0]
    for i in range(0, len(C)):
        new_x = R[0,0]*C[0][i] + R[0,1]*C[1][i] + cluster['Mean'][0]
        new_y = R[1,0]*C[0][i] + R[1,1]*C[1][i] + cluster['Mean'][1]
        C[0][i] = new_x
        C[1][i] = new_y
        if(new_x > x_lim[1]): x_lim[1] = new_x 
        if(new_x < x_lim[0]): x_lim[0] = new_x 
        if(new_y > y_lim[1]): y_lim[1] = new_y 
        if(new_y < y_lim[0]): y_lim[0] = new_y 
    
    verts = [*zip(C[0], C[1])]
    poly = Polygon(verts, facecolor=color, edgecolor=color)
    ax.add_patch(poly)
    return [x_lim, y_lim]


def plot_GMM(file, ax, color):
    data = get_json_from_file(file)
    
    lim=[[0,0],[0,0]]
    for k in range(0, len(data)):
        new_lim = plot_cluster(ax, data[k], color)
        if(new_lim[0][0] < lim[0][0]): lim[0][0] = new_lim[0][0]
        if(new_lim[0][1] > lim[0][1]): lim[0][1] = new_lim[0][1]
        if(new_lim[1][0] < lim[1][0]): lim[1][0] = new_lim[1][0]
        if(new_lim[1][1] > lim[1][1]): lim[1][1] = new_lim[1][1]
    return lim
    
fig = plt.figure()
ax = fig.gca()
lim = plot_GMM('learnt_model.json',ax,[0,1,0])
ax.set_xlim(lim[0][0], lim[0][1])
ax.set_ylim(lim[1][0], lim[1][1])
plt.show() 