# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:42:23 2019

@author: user
"""
import os
import random

import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from skimage.util.shape import view_as_blocks
from skimage.io import imread, imsave

import visdom







random.seed(0)

data_root = 'p3_data'
classes = os.listdir(data_root)

X_train = []
X_train_patches = []
X_train_labels = np.repeat([0, 1, 2, 3], 375)

X_test = []
X_test_patches = []
X_test_labels = np.repeat([0, 1, 2, 3], 125)

for c in classes:
    
    img_ls = os.listdir(os.path.join(data_root, c))
    random.shuffle(img_ls)
    
    X_train.append([])
    X_test.append([])
    
    for img in img_ls[:375]:
        
        I = imread(os.path.join(data_root, c, img))
        if len(I.shape) == 2:
            I = np.stack((I, I, I), axis=2)
        X_train[-1].append(I)
        
        I_patches = view_as_blocks(I, (16, 16, 3))
        X_train_patches.append(I_patches.reshape(16, -1))
        
    for img in img_ls[375:]:
        
        I = imread(os.path.join(data_root, c, img))
        if len(I.shape) == 2:
            I = np.stack((I, I, I), axis=2)
        X_test[-1].append(I)
        
        I_patches = view_as_blocks(I, (16, 16, 3))
        X_test_patches.append(I_patches.reshape(16, -1))

for i, img_ls in enumerate(X_train):
    imsave(classes[i]+'.png', img_ls[0])
    
    patches = view_as_blocks(img_ls[0], (16, 16, 3)) 
    for j in range(3):
        imsave(os.path.join(classes[i]+'{}.png'.format(j+1)), patches[0, j, 0])











X_train_patches = np.concatenate(X_train_patches, axis=0)
X_test_patches = np.concatenate(X_test_patches, axis=0)

pca = PCA(n_components=3)
X_train_patches_reduced = pca.fit_transform(X_train_patches)

kmeans = KMeans(n_clusters=15, random_state=0, max_iter=5000)
kmeans.fit(X_train_patches)
X_train_clusters = kmeans.labels_
index = np.where(X_train_clusters<6)[0]

vis = visdom.Visdom(env='hw1')
vis.scatter(X_train_patches_reduced[index], X_train_clusters[index]+1, win='PCA Subspace', opts={'title': 'PCA Subspace'})









X_train_bows = []
for i, features in enumerate(X_train_patches.reshape(1500, 16, -1)):
    
    features_dis = kmeans.transform(features)
    features_dis = np.reciprocal(features_dis)
    
    bow = []
    for dis in features_dis:
        dis = dis / dis.sum()
        bow.append(dis)
        
    bow = np.array(bow)
    bow = np.amax(bow, axis=0)
    
    X_train_bows.append(bow)
    
X_train_bows = np.array(X_train_bows)

rownames = [i+1 for i in range(15)]
for i in range(4):
    vis.bar(X_train_bows[375*i], win=classes[i], opts={'title':classes[i], 'rownames':rownames, 'ylabel':'value'})









    
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_bows, X_train_labels)

X_test_bows = []
for i, features in enumerate(X_test_patches.reshape(500, 16, -1)):
    
    features_dis = kmeans.transform(features)
    features_dis = np.reciprocal(features_dis)
    
    bow = []
    for dis in features_dis:
        dis = dis / dis.sum()
        bow.append(dis)
        
    bow = np.array(bow)
    bow = np.amax(bow, axis=0)
    
    X_test_bows.append(bow)
    
X_test_bows = np.array(X_test_bows)
    
print(knn.score(X_test_bows, X_test_labels))