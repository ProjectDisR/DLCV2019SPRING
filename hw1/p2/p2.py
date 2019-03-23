# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:26:08 2019

@author: user
"""

import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from skimage.io import imread, imsave




train_I = []
train_label = []
test_I = []
test_label = []

for i in range(1, 41):
    for j in range(1, 7):
        
        I = imread('p2_data/{}_{}.png'.format(i, j))
        I = np.array(I)
        I = np.reshape(I, -1)
        
        train_I.append(I)
        train_label.append(i)

train_I = np.array(train_I)
train_label = np.array(train_label)

for i in range(1, 41):
    for j in range(7, 11):
        
        I = imread('p2_data/{}_{}.png'.format(i, j))
        I = np.array(I)
        I = np.reshape(I, -1)
        
        test_I.append(I)
        test_label.append(i)
        
test_I = np.array(test_I)
test_label = np.array(test_label)

pca = PCA(n_components=239)
pca.fit(train_I)

for i, eigen in enumerate(pca.components_[:4]):
    
    eigen = eigen-np.min(eigen)
    eigen = (eigen/np.max(eigen)*255).astype(np.uint8)
    imsave('eigen{}.png'.format(i+1), eigen.reshape(56, 46)) 

mean = pca.mean_-np.min(pca.mean_)
mean = (mean/np.max(mean)*255).astype(np.uint8)
imsave('mean.png', mean.reshape(56, 46))












I = imread('p2_data/1_1.png')

for n in [3, 45, 140, 229]:
    
    pca = PCA(n_components=n)
    pca.fit(train_I)
    
    I_recons = pca.inverse_transform(pca.transform(np.reshape(I, (1, -1))))
    I_recons = np.reshape(I_recons, (56, 46))
    print(n, mean_squared_error(I, I_recons))  
    I_recons = np.uint8(I_recons)
    imsave('1_1eigen{}.png'.format(n), I_recons)












kfold = KFold(n_splits=3, shuffle=True, random_state=0)

for k in [1, 3, 5]:
    for n in [3, 45, 140]:
        
        accuracy = []
        
        for train_index, valid_index in kfold.split(train_I):
            
            train_I_split = train_I[train_index]
            train_label_split = train_label[train_index]
            valid_I_split = train_I[valid_index]
            valid_label_split = train_label[valid_index]
            
            pca = PCA(n_components=n)
            train_I_split_reduced = pca.fit_transform(train_I_split)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(train_I_split_reduced, train_label_split)
            
            valid_I_split_reduced = pca.transform(valid_I_split)
            accuracy.append(knn.score(valid_I_split_reduced, valid_label_split))
        
        print(k, n, accuracy[0], accuracy[1], accuracy[2], np.array(accuracy).mean())








pca = PCA(n_components=140)
train_I_reduced = pca.fit_transform(train_I)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_I_reduced, train_label)

test_I_reduced = pca.transform(test_I)
print(knn.score(test_I_reduced, test_label))