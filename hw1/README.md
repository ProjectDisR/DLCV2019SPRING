# hw 1: CV & ML Basics
CV & ML Basics [[pdf](hw1.pdf)] [[slides](hw1_intro.pdf)]
* Problem 1: Bayes Decision Rule
* Problem 2: Principal Component Analysis [[link](#principal-component-analysis)]
* Problem 3: Visual Bag-of-Words [[link](#visual-bag-of-Words)]
* Problem 4: Image Filtering [[link](#image-filtering)]

## Principal Component Analysis

### EigenFace
mean | eigen1 | eigen2 | eigen3 | eigen4
--- | --- | --- | --- | ---
![mean](p2/mean.png) | ![eigen1](p2/eigen1.png) | ![eigen2](p2/eigen2.png) | ![eigen3](p2/eigen3.png) | ![eigen4](p2/eigen4.png)
### Face Reconstruction
original | 3 eigens | 45 eigens | 140 eigens| 229 eigens
--- | --- | --- | --- | --- 
![original](p2/p2_data/1_1.png) <br/> MSE: 0 | ![3](p2/1_1eigen3.png) <br/> MSE: 1007 | ![45](p2/1_1eigen45.png) <br/> MSE: 277.3 | ![140](p2/1_1eigen140.png) <br/> MSE: 22.33 | ![229](p2/1_1eigen229.png) <br/> MSE: 0.1096

### PCA + KNN
Training Accuracy

k | n | fold1 | fold2 | fold3 | average
--- | --- | --- | --- | --- | ---
1 | 3 | 60.0% | 67.5% | 70.0% | 65.8%
1 | 45 | 90.0% | 86.3% | 90.0% | 88.8%
**1** | **140** | **90.0%** | **88.8%** | **91.3%** | **90.0%**
3 | 3 | 40.0% | 55.0% | 53.8% | 49.6%
3 | 45 | 73.8% | 81.3% | 68.8% | 74.6%
3 | 140 | 72.5% | 81.3% | 70.0% | 74.6%
5 | 3 | 38.8% | 47.5% | 43.8% | 43.3%
5 | 45 | 62.5% | 73.8% | 63.8% | 66.7%
5 | 140 | 58.8% | 73.8% | 58.8% | 63.8%

Testing Accuracy

k | n | Accuracy
--- | --- | ---
**1** | **140** | **94.375%**


## Visual Bag-of-Words

original | patch1 | patch2 | patch3
--- | --- | --- | ---
![banana](p3/banana.png) | ![banana1](p3/banana1.png) | ![banana2](p3/banana2.png) | ![banana3](p3/banana3.png)
![fountain](p3/fountain.png) | ![fountain1](p3/fountain1.png) | ![fountain2](p3/fountain2.png) | ![fountain3](p3/fountain3.png)
![reef](p3/reef.png) | ![reef1](p3/reef1.png) | ![reef2](p3/reef2.png) | ![reef3](p3/reef3.png)
![tractor](p3/tractor.png) | ![tractor1](p3/tractor1.png) | ![tractor2](p3/tractor2.png) | ![tractor3](p3/tractor3.png)

### PCA Subspace
![pca_subspace](p3/pca_subspace.png)

### BoW

banana | fountain | reef | tractor
--- | --- | --- | ---
![banana](p3/banana_bar.png) | ![fountain](p3/fountain_bar.png) | ![reef](p3/reef_bar.png) | ![tractor](p3/tractor_bar.png) 

### Testing
Accuracy: 55.6%

## Image Filtering
original | gaussian filtered | derivative_x | derivative_y
--- | --- | --- | ---
![original](p4/lena.png) | ![gaussian filtered](p4/lena_gaussian.png) | ![I_x](p4/lena_x.png) | ![I_y](p4/lena_y.png)

gradient of original | gradient of gaussian filtered
--- | ---
![gradient of original](p4/lena_grad.png) | ![gradient of gaussian filtered](p4/lena_gaussian_grad.png)
