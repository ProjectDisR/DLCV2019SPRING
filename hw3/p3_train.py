import os

import numpy as np

from sklearn.manifold import TSNE

import torch as t
from torch.utils.data import DataLoader
from torch import nn

from configs import DANNConfig
from datasets import USPS, MNISTM, SVHN, USPS2MNISTM, MNISTM2SVHN, SVHN2USPS
from models import DANN
from utils import AverageMeter

from tensorboardX import SummaryWriter
import visdom

import fire

device = t.device("cuda")

def train(**kwargs):
    
    opt = DANNConfig()
    opt.parse(kwargs)
    
    data_root = 'hw3_data/digits/'
    
    usps2mnistm = DataLoader(USPS2MNISTM(), opt.batch_size, shuffle=True)
    mnistm2svhn = DataLoader(MNISTM2SVHN(), opt.batch_size, shuffle=True)
    svhn2usps = DataLoader(SVHN2USPS(), opt.batch_size, shuffle=True)
    
    usps_dataset = USPS(data_root+'usps/', train=False)
    usps_test = DataLoader(usps_dataset, opt.batch_size, shuffle=False)
    
    mnistm_dataset = MNISTM(data_root+'mnistm/', train=False)
    mnistm_test = DataLoader(mnistm_dataset, opt.batch_size, shuffle=False)
    
    svhn_dataset = SVHN(data_root+'svhn/', train=False)
    svhn_test = DataLoader(svhn_dataset, opt.batch_size, shuffle=False)
    
    criterion_domain = nn.BCEWithLogitsLoss()
    criterion_class = nn.CrossEntropyLoss()
    
    writer = SummaryWriter('runs/dann/')
    vis = visdom.Visdom(env='dann', port=8888)

      
    dann = DANN().to(device)
    
    optimizer = t.optim.SGD(dann.parameters(), lr=opt.lr, momentum=0.9)
    
    total_steps = opt.n_epoch * len(usps2mnistm)
    
    if not os.path.isdir(os.path.join(opt.ckpts_root, 'mnistm/')):
        os.makedirs(os.path.join(opt.ckpts_root, 'mnistm/'))
        
    for epoch in range(opt.n_epoch):
        
        start_steps = epoch * len(usps2mnistm)
        
        loss_meter = AverageMeter()
        
        for i, (I1, labels1, I2) in enumerate(usps2mnistm):
            
            p = float(i + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr / (1. + 10 * p) ** 0.75
            
            
            I1 = I1.to(device)
            labels1 = labels1.to(device).long()
            I2 = I2.to(device)
            
            zeros = t.zeros(I1.size()[0], 1).to(device)
            ones = t.ones(I2.size()[0], 1).to(device)
            
            
            features, domains, classes_ = dann(I1, alpha)
            
            loss = criterion_domain(domains, zeros) + criterion_class(classes_, labels1) 
            
            features, domains, classes_ = dann(I2, alpha)
            
            loss += criterion_domain(domains, ones)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            loss_meter.update(loss.item(), I1.size()[0])
            
            print(i, loss.item())
        
        t.save(dann.state_dict(), os.path.join(opt.ckpts_root, 'mnistm/', 'e{}.ckpt'.format(epoch)))
        
        
        dann.eval() 
        
        feature_ls = []
        domain_ls = []
        label_ls = []
        
        n_correct = 0
        n = 0
    
        for i, (I, labels, img_names) in enumerate(mnistm_test):
                
            I = I.to(device)
            labels = labels.to(device).long()
            
            features, domains, classes_ = dann(I, 0)
            
            n_correct += (t.argmax(classes_, dim=1) == labels).sum().item()
            n += I.size()[0]
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 1000:
                
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.ones(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
        
        writer.add_scalar('acc_mnistm', n_correct/n, epoch)
        writer.add_scalar('loss_mnistm', loss_meter.avg, epoch)
        
        for i, (I, labels, img_names) in enumerate(usps_test):
                    
            I = I.to(device)
            labels = labels.to(device).long()
            
            features, domains, classes_ = dann(I, 0)
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 2000:
                
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.zeros(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
            
        tsne = TSNE().fit_transform(np.concatenate(feature_ls, axis=0))
        vis.scatter(tsne, np.concatenate(label_ls, axis=0)+1, win='mnistm_label', opts={'title':'mnistm_label', 'legend':[i for i in range(10)]})
        vis.scatter(tsne, np.concatenate(domain_ls, axis=0)+1, win='mnistm_domain', opts={'title': 'mnistm_domain', 'legend':[0, 1]})
        
        dann.train()


    dann = DANN().to(device)
    
    optimizer = t.optim.SGD(dann.parameters(), lr=opt.lr, momentum=0.9)
    
    total_steps = opt.n_epoch * len(mnistm2svhn)
    
    if not os.path.isdir(os.path.join(opt.ckpts_root, 'svhn/')):
        os.makedirs(os.path.join(opt.ckpts_root, 'svhn/'))
        
    for epoch in range(opt.n_epoch):
        
        start_steps = epoch * len(mnistm2svhn)
        
        loss_meter = AverageMeter()
        
        for i, (I1, labels1, I2) in enumerate(mnistm2svhn):
            
            p = float(i + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr / (1. + 10 * p) ** 0.75
            
            
            I1 = I1.to(device)
            labels1 = labels1.to(device).long()
            I2 = I2.to(device)
            
            zeros = t.zeros(I1.size()[0], 1).to(device)
            ones = t.ones(I2.size()[0], 1).to(device)
            
            
            features, domains, classes_ = dann(I1, alpha)
            
            loss = criterion_domain(domains, zeros) + criterion_class(classes_, labels1) 
            
            features, domains, classes_ = dann(I2, alpha)
            
            loss += criterion_domain(domains, ones)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            loss_meter.update(loss.item(), I1.size()[0])
            
            print(i, loss.item())
        
        t.save(dann.state_dict(), os.path.join(opt.ckpts_root, 'svhn/', 'e{}.ckpt'.format(epoch)))


        dann.eval() 
        
        feature_ls = []
        domain_ls = []
        label_ls = []
        
        n_correct = 0
        n = 0
    
        for i, (I, labels, img_names) in enumerate(svhn_test):
                
            I = I.to(device)
            labels = labels.to(device).long()
            
            features, domains, classes_ = dann(I, 0)
            
            n_correct += (t.argmax(classes_, dim=1) == labels).sum().item()
            n += I.size()[0]
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 1000:
                
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.ones(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
        
        writer.add_scalar('acc_svhn', n_correct/n, epoch)
        writer.add_scalar('loss_svhn', loss_meter.avg, epoch)
        
        for i, (I, labels, img_names) in enumerate(mnistm_test):
                    
            I = I.to(device)
            labels = labels.to(device).long()
            
            features, domains, classes_ = dann(I, 0)
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 2000:
                
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.zeros(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
            
        tsne = TSNE().fit_transform(np.concatenate(feature_ls, axis=0))
        vis.scatter(tsne, np.concatenate(label_ls, axis=0)+1, win='svhn_label', opts={'title':'svhn_label', 'legend':[i for i in range(10)]})
        vis.scatter(tsne, np.concatenate(domain_ls, axis=0)+1, win='svhn_domain', opts={'title': 'svhn_domain', 'legend':[0, 1]})
        
        dann.train()
        
    
    dann = DANN().to(device)
    
    optimizer = t.optim.SGD(dann.parameters(), lr=opt.lr, momentum=0.9)
    
    total_steps = opt.n_epoch * len(svhn2usps)
    
    if not os.path.isdir(os.path.join(opt.ckpts_root, 'usps/')):
        os.makedirs(os.path.join(opt.ckpts_root, 'usps/'))
        
    for epoch in range(opt.n_epoch):
        
        start_steps = epoch * len(svhn2usps)
        
        loss_meter = AverageMeter()
        
        for i, (I1, labels1, I2) in enumerate(svhn2usps):
            
            p = float(i + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr / (1. + 10 * p) ** 0.75
            
            
            I1 = I1.to(device)
            labels1 = labels1.to(device).long()
            I2 = I2.to(device)
            
            zeros = t.zeros(I1.size()[0], 1).to(device)
            ones = t.ones(I2.size()[0], 1).to(device)
            
            
            features, domains, classes_ = dann(I1, alpha)
            
            loss = criterion_domain(domains, zeros) + 16*criterion_class(classes_, labels1) 
            
            features, domains, classes_ = dann(I2, alpha)
            
            loss += criterion_domain(domains, ones)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            loss_meter.update(loss.item(), I1.size()[0])
            
            print(i, loss.item())
        
        t.save(dann.state_dict(), os.path.join(opt.ckpts_root, 'usps/', 'e{}.ckpt'.format(epoch)))


        dann.eval() 

        feature_ls = []
        domain_ls = []
        label_ls = []

        n_correct = 0
        n = 0

        for i, (I, labels, img_names) in enumerate(usps_test):
                
            I = I.to(device)
            labels = labels.to(device).long()
            
            features, domains, classes_ = dann(I, 0)
            
            n_correct += (t.argmax(classes_, dim=1) == labels).sum().item()
            n += I.size()[0]
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 1000:
                
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.ones(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
        
        writer.add_scalar('acc_usps', n_correct/n, epoch)
        writer.add_scalar('loss_usps', loss_meter.avg, epoch)

        for i, (I, labels, img_names) in enumerate(svhn_test):

            I = I.to(device)
            labels = labels.to(device).long()

            features, domains, classes_ = dann(I, 0)

            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 2000:
        
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.zeros(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())

        tsne = TSNE().fit_transform(np.concatenate(feature_ls, axis=0))
        vis.scatter(tsne, np.concatenate(label_ls, axis=0)+1, win='usps_label', opts={'title':'usps_label', 'legend':[i for i in range(10)]})
        vis.scatter(tsne, np.concatenate(domain_ls, axis=0)+1, win='usps_domain', opts={'title': 'usps_domain', 'legend':[0, 1]})

        dann.train()


if __name__ == '__main__':
    fire.Fire()