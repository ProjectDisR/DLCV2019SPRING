import os
import sys

import torch as t
import torchvision as tv
from torch.utils import data
from torch.utils.data import DataLoader

from models import Yolov1_vgg16bn

from config import BestConfig
from utils import NMS

from skimage.io import imread


device = t.device("cuda")


class AerialImages(data.Dataset):
    
    def __init__(self, root):
        
        self.root = root
        
        self.img_name_ls = os.listdir(os.path.join(self.root))
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


def test(imgs_root, predicts_root):
    
    opt = BestConfig()
    
    dataset = AerialImages(imgs_root)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False)

    yolov1 = Yolov1_vgg16bn(pretrained=True)
    yolov1.load_state_dict(t.load('e69.ckpt?dl=1'))
    yolov1 = yolov1.to(device).eval()
    
    if not os.path.isdir(predicts_root):
        os.makedirs(predicts_root)
    
    for I, img_names, (H, W) in dataloader:
        
        I = I.to(device)
        
        predicts = yolov1(I)
        predicts = predicts.detach()
        
        for n in range(predicts.size()[0]):
                
            final_indices, final_classes, final_scores = NMS(predicts[n], H[n], W[n], opt.thres, dataset.classes)
            
            with open(os.path.join(predicts_root, img_names[n].split('.')[0]+'.txt'), 'w') as predicttxt:
                
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
    
    test(sys.argv[1], sys.argv[2])