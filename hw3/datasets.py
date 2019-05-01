import os

import numpy as np

from torch.utils import data
import torchvision as tv

from skimage.io import imread

class Face(data.Dataset):
    
    def __init__(self, root):
        
        self.root = root
        
        self.img_name_ls = os.listdir(os.path.join(self.root, 'train/'))
        self.img_name_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
        
        self.attrs = np.zeros((len(self.img_name_ls), 1))
        
        with open(os.path.join(root, 'train.csv'), 'r') as attr_file:
            for i, line in enumerate(attr_file.readlines()):
                
                if i == 0: 
                    continue
                
                line = line.strip('/n').split(',')
                
                assert line[0] == self.img_name_ls[i-1]
                
                self.attrs[i-1, 0] = float(line[10])
    
        return
        
    def __getitem__(self, index):
        
        I = imread(os.path.join(self.root, 'train/', self.img_name_ls[index]))
        I = self.transforms(I)
        
        return I, self.attrs[index]
    
    def __len__(self):
        
        return len(self.img_name_ls)


class USPS(data.Dataset):
    
    def __init__(self, root, train):
        
        self.root = root
        
        self.folder = 'test/'
        if train:
            self.folder = 'train/'

        self.img_name_ls = os.listdir(os.path.join(self.root, self.folder))
        self.img_name_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    
    
        self.labels = np.zeros(len(self.img_name_ls))
        
        with open(os.path.join(root, self.folder[:-1]+'.csv'), 'r') as label_file:
            for i, line in enumerate(label_file.readlines()):
                
                if i == 0: 
                    continue
                
                line = line.strip('/n').split(',')
                
                assert line[0] == self.img_name_ls[i-1]
                
                self.labels[i-1] = float(line[1])
    
        return
        
    def __getitem__(self, index):
        
        I = imread(os.path.join(self.root, self.folder, self.img_name_ls[index]))
        I = np.stack([I, I, I], axis=2)
        I = self.transforms(I)
        
        return I, self.labels[index], self.img_name_ls[index]
    
    def __len__(self):
        
        return len(self.img_name_ls)


class MNISTM(data.Dataset):
    
    def __init__(self, root, train):
        
        self.root = root
        
        self.folder = 'test/'
        if train:
            self.folder = 'train/'

        self.img_name_ls = os.listdir(os.path.join(self.root, self.folder))
        self.img_name_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
    
        self.labels = np.zeros(len(self.img_name_ls))
        
        with open(os.path.join(root, self.folder[:-1]+'.csv'), 'r') as label_file:
            for i, line in enumerate(label_file.readlines()):
                
                if i == 0: 
                    continue
                
                line = line.strip('/n').split(',')
                
                assert line[0] == self.img_name_ls[i-1]
                
                self.labels[i-1] = float(line[1])
    
        return
        
    def __getitem__(self, index):
        
        I = imread(os.path.join(self.root, self.folder, self.img_name_ls[index]))
        I = self.transforms(I)
        
        return I, self.labels[index], self.img_name_ls[index]
    
    def __len__(self):
        
        return len(self.img_name_ls)


class SVHN(data.Dataset):
    
    def __init__(self, root, train):
        
        self.root = root    
        
        self.folder = 'test/'
        if train:
            self.folder = 'train/'

        self.img_name_ls = os.listdir(os.path.join(self.root, self.folder))
        self.img_name_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((28, 28)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
    
        self.labels = np.zeros(len(self.img_name_ls))
        
        with open(os.path.join(root, self.folder[:-1]+'.csv'), 'r') as label_file:
            for i, line in enumerate(label_file.readlines()):
                
                if i == 0: 
                    continue
                
                line = line.strip('/n').split(',')
                
                assert line[0] == self.img_name_ls[i-1]
                
                self.labels[i-1] = float(int(line[1]) % 10)
    
        return
        
    def __getitem__(self, index):
        
        I = imread(os.path.join(self.root, self.folder, self.img_name_ls[index]))
        I = self.transforms(I)
        
        return I, self.labels[index], self.img_name_ls[index]
    
    def __len__(self):
        
        return len(self.img_name_ls)


class USPS2MNISTM(data.Dataset):
    
    def __init__(self):
        
        self.img_name_ls1 = os.listdir('hw3_data/digits/usps/train/')
        self.img_name_ls1.sort()

        self.img_name_ls2 = os.listdir('hw3_data/digits/mnistm/train/')
        self.img_name_ls2.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
    
        self.labels1 = np.zeros(len(self.img_name_ls1))
        
        with open(os.path.join('hw3_data/digits/usps/', 'train.csv'), 'r') as label_file:
            for i, line in enumerate(label_file.readlines()):
                
                if i == 0: 
                    continue
                
                line = line.strip('/n').split(',')
                
                assert line[0] == self.img_name_ls1[i-1]
                
                self.labels1[i-1] = float(line[1])
    
        return
        
    def __getitem__(self, index):
        
        I1 = imread(os.path.join('hw3_data/digits/usps/train/', self.img_name_ls1[index % len(self.img_name_ls1)]))
        I1 = np.stack([I1, I1, I1], axis=2)
        I1 = self.transforms(I1)
        
        I2 = imread(os.path.join('hw3_data/digits/mnistm/train/', self.img_name_ls2[index]))
        I2 = self.transforms(I2)

        return I1, self.labels1[index % len(self.img_name_ls1)], I2
    
    def __len__(self):
        
        return len(self.img_name_ls2)


class MNISTM2SVHN(data.Dataset):
    
    def __init__(self):
        
        self.img_name_ls1 = os.listdir('hw3_data/digits/mnistm/train/')
        self.img_name_ls1.sort()

        self.img_name_ls2 = os.listdir('hw3_data/digits/svhn/train/')
        self.img_name_ls2.sort()
        
        self.transforms1 = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
        self.transforms2 = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((28, 28)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
    
        self.labels1 = np.zeros(len(self.img_name_ls1))
        
        with open(os.path.join('hw3_data/digits/mnistm/', 'train.csv'), 'r') as label_file:
            for i, line in enumerate(label_file.readlines()):
                
                if i == 0: 
                    continue
                
                line = line.strip('/n').split(',')
                
                assert line[0] == self.img_name_ls1[i-1]
                
                self.labels1[i-1] = float(line[1])
    
        return
        
    def __getitem__(self, index):
        
        I1 = imread(os.path.join('hw3_data/digits/mnistm/train/', self.img_name_ls1[index % len(self.img_name_ls1)]))
        I1 = self.transforms1(I1)
        
        I2 = imread(os.path.join('hw3_data/digits/svhn/train/', self.img_name_ls2[index]))
        I2 = self.transforms2(I2)
        
        return I1, self.labels1[index % len(self.img_name_ls1)], I2
    
    def __len__(self):
        
        return len(self.img_name_ls2)
    

class SVHN2USPS(data.Dataset):
    
    def __init__(self):
        
        self.img_name_ls1 = os.listdir('hw3_data/digits/svhn/train/')
        self.img_name_ls1.sort()

        self.img_name_ls2 = os.listdir('hw3_data/digits/usps/train/')
        self.img_name_ls2.sort()
        
        self.transforms1 = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((28, 28)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    
        self.transforms2 = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
    
        self.labels1 = np.zeros(len(self.img_name_ls1))
        
        with open(os.path.join('hw3_data/digits/svhn/', 'train.csv'), 'r') as label_file:
            for i, line in enumerate(label_file.readlines()):
                
                if i == 0: 
                    continue
                
                line = line.strip('/n').split(',')
                
                assert line[0] == self.img_name_ls1[i-1]
                
                self.labels1[i-1] = float(int(line[1]) % 10)
    
        return
        
    def __getitem__(self, index):
        
        I1 = imread(os.path.join('hw3_data/digits/svhn/train/', self.img_name_ls1[index]))
        I1 = self.transforms1(I1)
        
        I2 = imread(os.path.join('hw3_data/digits/usps/train/', self.img_name_ls2[index % len(self.img_name_ls2)]))
        I2 = np.stack([I2, I2, I2], axis=2)
        I2 = self.transforms2(I2)
        
        return I1, self.labels1[index], I2
    
    def __len__(self):
        
        return len(self.img_name_ls1)