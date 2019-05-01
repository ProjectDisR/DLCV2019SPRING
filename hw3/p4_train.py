import os

import numpy as np

from sklearn.manifold import TSNE

import torch as t
from torch.utils.data import DataLoader
from torch import nn

from configs import ADDAConfig
from datasets import USPS, MNISTM, SVHN, USPS2MNISTM, MNISTM2SVHN, SVHN2USPS
from models import CNN, Classifier, Domain_Discriminator

from tensorboardX import SummaryWriter
import visdom

import fire

device = t.device("cuda")

def train(**kwargs):
    
    opt = ADDAConfig()
    opt.parse(kwargs)
    
    data_root = 'hw3_data/digits/'
    
    usps2mnistm = DataLoader(USPS2MNISTM(), opt.batch_size, shuffle=True)
    mnistm2svhn = DataLoader(MNISTM2SVHN(), opt.batch_size, shuffle=True)
    svhn2usps = DataLoader(SVHN2USPS(), opt.batch_size, shuffle=True)
     
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
    criterion_domain = nn.BCEWithLogitsLoss()
    
    writer = SummaryWriter('runs/adda/')
    vis = visdom.Visdom(env='adda', port=8888)

  
    cnn1 = CNN().to(device)
    classifier = Classifier().to(device)
    
    optimizer = t.optim.SGD(list(cnn1.parameters())+list(classifier.parameters()), lr=0.01, momentum=0.9)
    
    if not os.path.isdir(os.path.join(opt.ckpts_root, 'mnistm/')):
        os.makedirs(os.path.join(opt.ckpts_root, 'mnistm/'))
    
    for epoch in range(30):
        
        for i, (I, labels, img_names) in enumerate(usps_train):
            
            I = I.to(device)
            labels = labels.to(device).long()
            
            features = cnn1(I)
            classes_ = classifier(features)
            
            loss = criterion_class(classes_, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    t.save(classifier.state_dict(), os.path.join(opt.ckpts_root, 'mnistm/', 'classifier.ckpt'))
    
    cnn1.eval()
    classifier.eval()
    
    cnn2 = CNN().to(device)
    cnn2.load_state_dict(cnn1.state_dict())
    discriminator = Domain_Discriminator().to(device)
    
    optimizer_cnn2 = t.optim.Adam(cnn2.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_discriminator = t.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    for epoch in range(opt.n_epoch):
        
        for i, (I1, labels1, I2)in enumerate(usps2mnistm):
            
            I1 = I1.to(device)
            I2 = I2.to(device)       
            
            features1 = cnn1(I1).detach()
            features2 = cnn2(I2)
            
            loss_cnn2 = criterion_domain(discriminator(features2), t.zeros(I2.size()[0], 1).to(device))
            
            optimizer_cnn2.zero_grad()
            loss_cnn2.backward()
            optimizer_cnn2.step()
            
            features2 = features2.detach()
            
            loss_discriminator = criterion_domain(discriminator(features1), t.zeros(I1.size()[0], 1).to(device)) + criterion_domain(discriminator(features2), t.ones(I2.size()[0], 1).to(device))
            
            optimizer_discriminator.zero_grad()
            loss_discriminator.backward()
            optimizer_discriminator.step()
            
        t.save(cnn2.state_dict(), os.path.join(opt.ckpts_root, 'mnistm/', 'e{}.ckpt'.format(epoch)))
        
        cnn2.eval()  
        
        feature_ls = []
        domain_ls = []
        label_ls = []
        
        n_correct = 0
        n = 0
    
        for i, (I, labels, img_names) in enumerate(mnistm_test):
                
            I = I.to(device)
            labels = labels.to(device).long()
            
            features = cnn2(I)
            classes_ = classifier(features)
            
            n_correct += (t.argmax(classes_, dim=1) == labels).sum().item()
            n += I.size()[0]
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 1000:
            
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.ones(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
                
        writer.add_scalar('acc_mnistm', n_correct/n, epoch)
        
        for i, (I, labels, img_names) in enumerate(usps_test):
                    
            I = I.to(device)
            labels = labels.to(device).long()
                
            features = cnn1(I)
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 2000:
                
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.zeros(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
                
        tsne = TSNE().fit_transform(np.concatenate(feature_ls, axis=0))
        vis.scatter(tsne, np.concatenate(label_ls, axis=0)+1, win='mnistm_label', opts={'title':'mnistm_label', 'legend':[i for i in range(10)]})
        vis.scatter(tsne, np.concatenate(domain_ls, axis=0)+1, win='mnistm_domain', opts={'title': 'mnistm_domain', 'legend':[0, 1]})
        
        cnn2.train()


    cnn1 = CNN().to(device)
    classifier = Classifier().to(device)
    
    optimizer = t.optim.SGD(list(cnn1.parameters())+list(classifier.parameters()), lr=0.01, momentum=0.9)
    
    if not os.path.isdir(os.path.join(opt.ckpts_root, 'svhn/')):
        os.makedirs(os.path.join(opt.ckpts_root, 'svhn/'))
    
    for epoch in range(30):
        
        for i, (I, labels, img_names) in enumerate(mnistm_train):
            
            I = I.to(device)
            labels = labels.to(device).long()
            
            features = cnn1(I)
            classes_ = classifier(features)
            
            loss = criterion_class(classes_, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    t.save(classifier.state_dict(), os.path.join(opt.ckpts_root, 'svhn/', 'classifier.ckpt'))
    
    cnn1.eval()
    classifier.eval()
    
    cnn2 = CNN().to(device)
    cnn2.load_state_dict(cnn1.state_dict())
    discriminator = Domain_Discriminator().to(device)
    
    optimizer_cnn2 = t.optim.Adam(cnn2.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_discriminator = t.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    for epoch in range(opt.n_epoch):
        
        for i, (I1, labels1, I2)in enumerate(mnistm2svhn):
            
            I1 = I1.to(device)
            I2 = I2.to(device)       
            
            features1 = cnn1(I1).detach()
            features2 = cnn2(I2)
            
            loss_cnn2 = criterion_domain(discriminator(features2), t.zeros(I2.size()[0], 1).to(device))
            
            optimizer_cnn2.zero_grad()
            loss_cnn2.backward()
            optimizer_cnn2.step()
            
            features2 = features2.detach()
            
            loss_discriminator = criterion_domain(discriminator(features1), t.zeros(I1.size()[0], 1).to(device)) + criterion_domain(discriminator(features2), t.ones(I2.size()[0], 1).to(device))
            
            optimizer_discriminator.zero_grad()
            loss_discriminator.backward()
            optimizer_discriminator.step()
            
        t.save(cnn2.state_dict(), os.path.join(opt.ckpts_root, 'svhn/', 'e{}.ckpt'.format(epoch)))
        
        cnn2.eval()  
        
        feature_ls = []
        domain_ls = []
        label_ls = []
        
        n_correct = 0
        n = 0
    
        for i, (I, labels, img_names) in enumerate(svhn_test):
                
            I = I.to(device)
            labels = labels.to(device).long()
            
            features = cnn2(I)
            classes_ = classifier(features)
            
            n_correct += (t.argmax(classes_, dim=1) == labels).sum().item()
            n += I.size()[0]
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 1000:
            
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.ones(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
                
        writer.add_scalar('acc_svhn', n_correct/n, epoch)
        
        for i, (I, labels, img_names) in enumerate(mnistm_test):
                    
            I = I.to(device)
            labels = labels.to(device).long()
                
            features = cnn1(I)
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 2000:
                
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.zeros(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
                
        tsne = TSNE().fit_transform(np.concatenate(feature_ls, axis=0))
        vis.scatter(tsne, np.concatenate(label_ls, axis=0)+1, win='svhn_label', opts={'title':'svhn_label', 'legend':[i for i in range(10)]})
        vis.scatter(tsne, np.concatenate(domain_ls, axis=0)+1, win='svhn_domain', opts={'title': 'svhn_domain', 'legend':[0, 1]})
        
        cnn2.train()


    cnn1 = CNN().to(device)
    classifier = Classifier().to(device)
    
    optimizer = t.optim.SGD(list(cnn1.parameters())+list(classifier.parameters()), lr=0.01, momentum=0.9)
    
    if not os.path.isdir(os.path.join(opt.ckpts_root, 'usps/')):
        os.makedirs(os.path.join(opt.ckpts_root, 'usps/'))
    
    for epoch in range(30):
        
        for i, (I, labels, img_names) in enumerate(svhn_train):
            
            I = I.to(device)
            labels = labels.to(device).long()
            
            features = cnn1(I)
            classes_ = classifier(features)
            
            loss = criterion_class(classes_, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    t.save(classifier.state_dict(), os.path.join(opt.ckpts_root, 'usps/', 'classifier.ckpt'))
    
    cnn1.eval()
    classifier.eval()
    
    cnn2 = CNN().to(device)
    cnn2.load_state_dict(cnn1.state_dict())
    discriminator = Domain_Discriminator().to(device)
    
    optimizer_cnn2 = t.optim.Adam(cnn2.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_discriminator = t.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    for epoch in range(opt.n_epoch):
        
        for i, (I1, labels1, I2)in enumerate(svhn2usps):
            
            I1 = I1.to(device)
            I2 = I2.to(device)       
            
            features1 = cnn1(I1).detach()
            features2 = cnn2(I2)
            
            loss_cnn2 = criterion_domain(discriminator(features2), t.zeros(I2.size()[0], 1).to(device))
            
            optimizer_cnn2.zero_grad()
            loss_cnn2.backward()
            optimizer_cnn2.step()
            
            features2 = features2.detach()
            
            loss_discriminator = criterion_domain(discriminator(features1), t.zeros(I1.size()[0], 1).to(device)) + criterion_domain(discriminator(features2), t.ones(I2.size()[0], 1).to(device))
            
            optimizer_discriminator.zero_grad()
            loss_discriminator.backward()
            optimizer_discriminator.step()
            
        t.save(cnn2.state_dict(), os.path.join(opt.ckpts_root, 'usps/', 'e{}.ckpt'.format(epoch)))
        
        cnn2.eval()  
        
        feature_ls = []
        domain_ls = []
        label_ls = []
        
        n_correct = 0
        n = 0
    
        for i, (I, labels, img_names) in enumerate(usps_test):
                
            I = I.to(device)
            labels = labels.to(device).long()
            
            features = cnn2(I)
            classes_ = classifier(features)
            
            n_correct += (t.argmax(classes_, dim=1) == labels).sum().item()
            n += I.size()[0]
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 1000:
            
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.ones(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
                
        writer.add_scalar('acc_usps', n_correct/n, epoch)
        
        for i, (I, labels, img_names) in enumerate(svhn_test):
                    
            I = I.to(device)
            labels = labels.to(device).long()
                
            features = cnn1(I)
            
            if i == 0 or len(np.concatenate(feature_ls, axis=0)) < 2000:
                
                feature_ls.append(features.detach().cpu().numpy())
                domain_ls.append(t.zeros(I.size()[0]).long().detach().cpu().numpy())
                label_ls.append(labels.detach().cpu().numpy())
                
        tsne = TSNE().fit_transform(np.concatenate(feature_ls, axis=0))
        vis.scatter(tsne, np.concatenate(label_ls, axis=0)+1, win='usps_label', opts={'title':'usps_label', 'legend':[i for i in range(10)]})
        vis.scatter(tsne, np.concatenate(domain_ls, axis=0)+1, win='usps_domain', opts={'title': 'usps_domain', 'legend':[0, 1]})
        
        cnn2.train()


if __name__ == '__main__':
    fire.Fire()