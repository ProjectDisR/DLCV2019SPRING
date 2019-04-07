import os

import torch as t
import torchvision as tv
from torch.utils import data

from skimage.io import imread


class AerialImages(data.Dataset):
    
    def __init__(self, root):
        
        self.root = root
        
        self.img_name_ls = os.listdir(os.path.join(self.root, 'images/'))
        self.img_name_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((448, 448)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
        
        self.classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
        
        self.labels = dict()
        
        for i, c in enumerate(self.classes):
            self.labels[c] = i
    
        return
        
    def __getitem__(self, index):
        
        img_name = self.img_name_ls[index]
        
        I = imread(os.path.join(self.root, 'images/', img_name))
        
        H = I.shape[0]
        W = I.shape[1]
        
        I = self.transforms(I)
        
        
        label = t.zeros(26, 7, 7)
        
        with open(os.path.join(self.root, 'labelTxt_hbb/', img_name.split('.')[0]+'.txt')) as labeltxt:
            
            for line in labeltxt.readlines():
                
                line = line.strip('\n').split(' ')
                
                x = (float(line[0]) + float(line[4])) / 2.0
                y = (float(line[1]) + float(line[5])) / 2.0
                w = float(line[4]) - float(line[0])
                h = float(line[5]) - float(line[1])
                
                x = x / (W-1) * 447.0
                y = y / (H-1) * 447.0
                w = w / (W-1) * 447.0
                h = h / (H-1) * 447.0
                
                
                i = int(min(max(y//64, 0), 6))
                j = int(min(max(x//64, 0), 6))
                
                label[0:10:5, i ,j] = (x - j*64.0) / 64.0
                label[1:10:5, i, j] = (y - i*64.0) / 64.0
                label[2:10:5, i, j] = w / 447.0
                label[3:10:5, i, j] = h / 447.0
                label[4:10:5, i, j] = 1.0
                label[10:, i, j]= 0.0
                label[self.labels[line[8]]+10, i, j]= 1.0
        
        return I, label, img_name, (H, W)
    
    def __len__(self):
        
        return len(self.img_name_ls)