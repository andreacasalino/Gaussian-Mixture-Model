# -*- coding: utf-8 -*-
"""
 * Author:    Andrea Casalino
 * Created:   26.12.2019
*
* report any bug to andrecasa91@gmail.com.
"""
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.spatial import ConvexHull


def get_json_from_file(name):
    with open(name) as json_file:
        return json.load(json_file)

def get_R(covariance_matrix):
    m = np.matrix(covariance_matrix)
    E, R = np.linalg.eig(m)
    for i in range(0, len(E)) :
        R[:,i] = R[:,i] * np.sqrt(E[i]) 
    return R
       
def plot_facet(x, y, z, ax, col, alp):
    ax.add_collection3d(Poly3DCollection([list(zip(x,y,z))], edgecolor=col, facecolors=col, alpha=alp, linewidth=0))
 
def plot_CH(Vertices, color, ax, alp):  
    Vertices_array = np.array(Vertices)
    hull = ConvexHull(Vertices_array)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        plot_facet(Vertices_array[s, 0], Vertices_array[s, 1], Vertices_array[s, 2], ax, color, alp)
        ax.plot(Vertices_array[s, 0], Vertices_array[s, 1], Vertices_array[s, 2] , '.', color=color, markersize=0.001)

def plot_cluster(ax, cluster, color):
       
    psi = np.linspace(0, 2 * np.pi, num=20)
    teta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, num=10)
    C = []
    R = get_R(cluster['Covariance']);
    
    for p in psi:
        for t in teta:
            V = [np.cos(p)*np.cos(t), np.sin(p)*np.cos(t), np.sin(t)]
            new_x = R[0,0] * V[0] + R[0,1] * V[1] + R[0,2] * V[2] +  cluster['Mean'][0]
            new_y = R[1,0] * V[0] + R[1,1] * V[1] + R[1,2] * V[2] +  cluster['Mean'][1]
            new_z = R[2,0] * V[0] + R[2,1] * V[1] + R[2,2] * V[2] +  cluster['Mean'][2]
            C.append([new_x,new_y,new_z])
    
    plot_CH(C, color, ax, cluster['w']);
    ax.plot([cluster['Mean'][0]] ,[cluster['Mean'][1]], [cluster['Mean'][2]] , '*k', markersize=4)
    ax.text(cluster['Mean'][0] ,cluster['Mean'][1], cluster['Mean'][2] , r"$w$=" + str(cluster['w']), color='black')
    


def plot_GMM(file, ax, color):
    data = get_json_from_file(file)
    
    for k in range(0, len(data)):
        plot_cluster(ax, data[k], color)
    
fig = plt.figure()
ax = fig.gca(projection='3d')
lim = plot_GMM('random_model.json',ax,[0,1,0])
plt.title('real model cluster covariances (trasparency proportional to the clusters weigths)')
plt.show() 

fig = plt.figure()
ax = fig.gca(projection='3d')
lim = plot_GMM('learnt_model.json',ax,[0,1,0])
plt.title('learnt model cluster covariances (trasparency proportional to the clusters weigths)')
plt.show() 