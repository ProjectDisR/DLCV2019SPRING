# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:06:41 2019

@author: user
"""

import numpy as np
import cv2

from skimage.io import imread, imsave






I = imread('lena.png')
I_gaussian = cv2.GaussianBlur(I, (3, 3), 0.7213)
imsave('lena_gaussian.png', I_gaussian)







I_x = (I[:, 2:] - I[:, :-2])//2
I_y = (I[2:, :] - I[:-2, :])//2
imsave('lena_x.png', I_x)
imsave('lena_y.png', I_y)








I_x = I_x.astype(np.float16)
I_y = I_y.astype(np.float16)
I_grad = (I_x[1:-1, :]**2+I_y[:, 1:-1]**2)**0.5
I_grad = I_grad-np.min(I_grad)
I_grad = (I_grad/np.max(I_grad)*255).astype(np.uint8)

I_gaussian_x = (I_gaussian[:, 2:] - I_gaussian[:, :-2])//2
I_gaussian_y = (I_gaussian[2:, :] - I_gaussian[:-2, :])//2
I_gaussian_x = I_gaussian_x.astype(np.float16)
I_gaussian_y = I_gaussian_y.astype(np.float16)
I_gaussian_grad = (I_gaussian_x[1:-1, :]**2+I_gaussian_y[:, 1:-1]**2)**0.5
I_gaussian_grad = I_gaussian_grad-np.min(I_gaussian_grad)
I_gaussian_grad = (I_gaussian_grad/np.max(I_gaussian_grad)*255).astype(np.uint8)

imsave('lena_grad.png', I_grad)
imsave('lena_gaussian_grad.png', I_gaussian_grad)