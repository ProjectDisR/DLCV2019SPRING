import os
import sys

import numpy as np

import torch as t
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision as tv

from configs import DANNConfig
from models import DANN

from skimage.io import imread

device = t.device("cuda")

class USPS(data.Dataset):
    
    def __init__(self, root):
        
        self.root = root

        self.img_name_ls = os.listdir(self.root)
        self.img_name_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    
        return
        
    def __getitem__(self, index):
        
        I = imread(os.path.join(self.root, self.img_name_ls[index]))
        I = np.stack([I, I, I], axis=2)
        I = self.transforms(I)
        
        return I, self.img_name_ls[index]
    
    def __len__(self):
        
        return len(self.img_name_ls)


class MNISTM(data.Dataset):
    
    def __init__(self, root):
        
        self.root = root

        self.img_name_ls = os.listdir(self.root)
        self.img_name_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    
        return
        
    def __getitem__(self, index):
        
        I = imread(os.path.join(self.root, self.img_name_ls[index]))
        I = self.transforms(I)
        
        return I, self.img_name_ls[index]
    
    def __len__(self):
        
        return len(self.img_name_ls)


class SVHN(data.Dataset):
    
    def __init__(self, root):
        
        self.root = root

        self.img_name_ls = os.listdir(self.root)
        self.img_name_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((28, 28)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    
        return
        
    def __getitem__(self, index):
        
        I = imread(os.path.join(self.root, self.img_name_ls[index]))
        I = self.transforms(I)
        
        return I, self.img_name_ls[index]
    
    def __len__(self):
        
        return len(self.img_name_ls)


opt = DANNConfig()

dann = DANN()

if sys.argv[2] == 'mnistm':
    
    mnistm_dataset = MNISTM(sys.argv[1])        
    test = DataLoader(mnistm_dataset, opt.batch_size, shuffle=False)
    
    dann.load_state_dict(t.load(os.path.join(opt.ckpts_root, 'mnistm/e4.ckpt')))
    
elif sys.argv[2] == 'svhn':
    
    svhn_dataset = SVHN(sys.argv[1])        
    test = DataLoader(svhn_dataset, opt.batch_size, shuffle=False)
    
    dann.load_state_dict(t.load(os.path.join(opt.ckpts_root, 'svhn/e45.ckpt')))
    
elif sys.argv[2] == 'usps':
    
    usps_dataset = USPS(sys.argv[1])        
    test = DataLoader(usps_dataset, opt.batch_size, shuffle=False)
    
    dann.load_state_dict(t.load(os.path.join(opt.ckpts_root, 'usps/e12.ckpt')))


dann = dann.eval().to(device)

with open(sys.argv[3], 'w') as predict_file:
    
    for I, img_names in test:
        
        I = I.to(device)
        
        features, domains, classes_ = dann(I, 0)
 
        for img_name, class_ in zip(img_names, classes_):
 
            predict_file.write(img_name+',{}\n'.format(t.argmax(class_, dim=0).item()))