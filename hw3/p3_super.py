import torch as t
from torch.utils.data import DataLoader
from torch import nn

from configs import DANNConfig
from datasets import USPS, MNISTM, SVHN
from models import DANN
from utils import AverageMeter

from tensorboardX import SummaryWriter

import fire

device = t.device("cuda")

def train(**kwargs):
    
    opt = DANNConfig()
    opt.parse(kwargs)
    
    data_root = 'hw3_data/digits/'
     
    usps_dataset = USPS(data_root+'usps/', train=True)
    usps_train = DataLoader(usps_dataset, opt.batch_size, shuffle=True) 
    usps_dataset = USPS(data_root+'usps/', train=False)
    usps_test = DataLoader(usps_dataset, opt.batch_size, shuffle=False)
    
    mnistm_dataset = MNISTM(data_root+'mnistm/', train=True)
    mnistm_train = DataLoader(mnistm_dataset, opt.batch_size, shuffle=True) 
    mnistm_dataset = MNISTM(data_root+'mnistm/', train=False)
    mnistm_test = DataLoader(mnistm_dataset, opt.batch_size, shuffle=False)
    
    svhn_dataset = SVHN(data_root+'svhn/', train=True)
    svhn_train = DataLoader(svhn_dataset, opt.batch_size, shuffle=True) 
    svhn_dataset = SVHN(data_root+'svhn/', train=False)
    svhn_test = DataLoader(svhn_dataset, opt.batch_size, shuffle=False)
    
    criterion_class = nn.CrossEntropyLoss()
    
    writer = SummaryWriter('runs/super/')
    
    
    dann = DANN().to(device)
    
    optimizer = t.optim.SGD(dann.parameters(), lr=opt.lr, momentum=0.9)
    
    for epoch in range(opt.n_epoch):
        
        loss_meter = AverageMeter()
        
        for i, (I, labels, img_names) in enumerate(mnistm_train):
            
            I = I.to(device)
            labels = labels.to(device).long()
            
            features, domains, classes_ = dann(I, 0)
            
            loss = criterion_class(classes_, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), I.size()[0])
        
        n_correct = 0
        n = 0
    
        for i, (I, labels, img_names) in enumerate(mnistm_test):
                
                I = I.to(device)
                labels = labels.to(device).long()
                
                features, domains, classes_ = dann(I, 0)
                
                n_correct += (t.argmax(classes_, dim=1) == labels).sum().item()
                n += I.size()[0]
                
        writer.add_scalar('acc_mnistm', n_correct/n, epoch)
        writer.add_scalar('loss_mnistm', loss_meter.avg, epoch)
        
    
    dann = DANN().to(device)
    
    optimizer = t.optim.SGD(dann.parameters(), lr=opt.lr, momentum=0.9)
    
    for epoch in range(opt.n_epoch):
        
        loss_meter = AverageMeter()
        
        for i, (I, labels, img_names) in enumerate(svhn_train):
            
            I = I.to(device)
            labels = labels.to(device).long()
            
            features, domains, classes_ = dann(I, 0)
            
            loss = criterion_class(classes_, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), I.size()[0])
        
        n_correct = 0
        n = 0
    
        for i, (I, labels, img_names) in enumerate(svhn_test):
                
                I = I.to(device)
                labels = labels.to(device).long()
                
                features, domains, classes_ = dann(I, 0)
                
                n_correct += (t.argmax(classes_, dim=1) == labels).sum().item()
                n += I.size()[0]
                
        writer.add_scalar('acc_svhn', n_correct/n, epoch)
        writer.add_scalar('loss_svhn', loss_meter.avg, epoch)
        
    
    dann = DANN().to(device)
    
    optimizer = t.optim.SGD(dann.parameters(), lr=opt.lr, momentum=0.9)
    
    for epoch in range(opt.n_epoch):
        
        loss_meter = AverageMeter()
        
        for i, (I, labels, img_names) in enumerate(usps_train):
            
            I = I.to(device)
            labels = labels.to(device).long()
            
            features, domains, classes_ = dann(I, 0)
            
            loss = criterion_class(classes_, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), I.size()[0])
        
        n_correct = 0
        n = 0
    
        for i, (I, labels, img_names) in enumerate(usps_test):
                
                I = I.to(device)
                labels = labels.to(device).long()
                
                features, domains, classes_ = dann(I, 0)
                
                n_correct += (t.argmax(classes_, dim=1) == labels).sum().item()
                n += I.size()[0]
                
        writer.add_scalar('acc_usps', n_correct/n, epoch)
        writer.add_scalar('loss_usps', loss_meter.avg, epoch)


if __name__ == '__main__':
    fire.Fire()