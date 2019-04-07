import os

import torch as t
import torchvision as tv
from torch.utils import data
from torch.utils.data import DataLoader

from models import Yolov1_vgg16bn

from config import BaselineConfig, BestConfig
from utils import NMS

from skimage.io import imread


device = t.device("cuda")


class AerialImages(data.Dataset):
    
    def __init__(self):
        
        self.root = 'hw2_train_val/val1500/images/'
        
        self.img_name_ls = ['0076.jpg', '0086.jpg', '0907.jpg']
        self.img_name_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((448, 448)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
        
        self.classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
    
        return
        
    def __getitem__(self, index):
        
        img_name = self.img_name_ls[index]
        
        I = imread(os.path.join(self.root, img_name))
        
        H = I.shape[0]
        W = I.shape[1]
        
        I = self.transforms(I)
        
        return I, img_name, (H, W)
    
    def __len__(self):
        
        return len(self.img_name_ls)
    
def report():
    
    predicts_root = 'baseline/'
    
    dataset = AerialImages()
    dataloader = DataLoader(dataset, 3, shuffle=False)

    yolov1 = Yolov1_vgg16bn(pretrained=True)
    
    if not os.path.isdir(predicts_root):
        os.makedirs(predicts_root)
    
    for s, e in enumerate([1, 28, 89]):
        
        yolov1.load_state_dict(t.load('ckpts/baseline/e{}.ckpt'.format(e)))
        yolov1 = yolov1.to(device).eval()
        
        for I, img_names, (H, W) in dataloader:
            
            I = I.to(device)
            
            predicts = yolov1(I)
            predicts = predicts.detach()
            
            for n in range(predicts.size()[0]):
                
                final_indices, final_classes, final_scores = NMS(predicts[n], H[n], W[n], 0.1, dataset.classes)
                        
                stage = ''    
                
                if s == 0:
                    stage = 'early'
                elif s == 1:
                    stage = 'middle'
                else:
                    stage = 'final'
                    
                with open(os.path.join(predicts_root, stage+img_names[n].split('.')[0]+'.txt'), 'w') as predicttxt:
                    
                    for index, class_, score in zip(final_indices, final_classes, final_scores):
                    
                        x = index[0]
                        y = index[1]
                        w = index[2]
                        h = index[3]
                        
                        
                        predicttxt.write(str(x-w/2) + ' ')
                        predicttxt.write(str(y-h/2) + ' ')
                        
                        predicttxt.write(str(x+w/2) + ' ')
                        predicttxt.write(str(y-h/2) + ' ')
                        
                        predicttxt.write(str(x+w/2) + ' ')
                        predicttxt.write(str(y+h/2) + ' ')
                        
                        predicttxt.write(str(x-w/2) + ' ')
                        predicttxt.write(str(y+h/2) + ' ')
                        
                        predicttxt.write(class_ + ' ')
                        predicttxt.write(str(score) + '\n')

    predicts_root = 'best/'
    
    dataset = AerialImages()
    dataloader = DataLoader(dataset, 3, shuffle=False)

    yolov1 = Yolov1_vgg16bn(pretrained=True)
    
    if not os.path.isdir(predicts_root):
        os.makedirs(predicts_root)
    
    for s, e in enumerate([1, 29, 69]):
        
        yolov1.load_state_dict(t.load('ckpts/best/e{}.ckpt'.format(e)))
        yolov1 = yolov1.to(device).eval()
        
        for I, img_names, (H, W) in dataloader:
            
            I = I.to(device)
            
            predicts = yolov1(I)
            predicts = predicts.detach()
            
            for n in range(predicts.size()[0]):
                
                final_indices, final_classes, final_scores = NMS(predicts[n], H[n], W[n], 0.1, dataset.classes)
                        
                stage = ''    
                
                if s == 0:
                    stage = 'early'
                elif s == 1:
                    stage = 'middle'
                else:
                    stage = 'final'
                    
                with open(os.path.join(predicts_root, stage+img_names[n].split('.')[0]+'.txt'), 'w') as predicttxt:
                    
                    for index, class_, score in zip(final_indices, final_classes, final_scores):
                    
                        x = index[0]
                        y = index[1]
                        w = index[2]
                        h = index[3]
                        
                        
                        predicttxt.write(str(x-w/2) + ' ')
                        predicttxt.write(str(y-h/2) + ' ')
                        
                        predicttxt.write(str(x+w/2) + ' ')
                        predicttxt.write(str(y-h/2) + ' ')
                        
                        predicttxt.write(str(x+w/2) + ' ')
                        predicttxt.write(str(y+h/2) + ' ')
                        
                        predicttxt.write(str(x-w/2) + ' ')
                        predicttxt.write(str(y+h/2) + ' ')
                        
                        predicttxt.write(class_ + ' ')
                        predicttxt.write(str(score) + '\n')
   
    return

if __name__ == '__main__':
    
    report()