import os

import torch as t
from torch.utils.data import DataLoader
from torch import nn
import torchvision as tv

from configs import ACGANConfig
from datasets import Face
from models import Generator, Discriminator
from utils import AverageMeter

from tensorboardX import SummaryWriter

import fire

device = t.device("cuda")

def train(**kwargs):
    
    opt = ACGANConfig()
    opt.parse(kwargs)
    
    data_root = 'hw3_data/face/'
    
    train_dataset = Face(data_root)
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=True)
    
    G = Generator().to(device)
    D = Discriminator().to(device)
    
    criterion_real = nn.BCEWithLogitsLoss()
    criterion_attr = nn.BCEWithLogitsLoss()
    
    optimizerG = t.optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerD = t.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    writer = SummaryWriter('runs/acgan/')
    
    if not os.path.isdir(opt.ckpts_root):
        os.makedirs(opt.ckpts_root)
    
    attr = t.zeros(20, 1).to(device)
    attr[0:10, 0] = 1.0
    
    noises = t.randn(10, 100).to(device)
    noises = noises.repeat(2, 1)
    
    fix_noises = t.cat((noises, attr), dim=1)
    
    for epoch in range(opt.n_epoch):
        
        lossG_meter = AverageMeter()
        lossD_meter = AverageMeter()
        
        for i, (realI, attr_labels) in enumerate(train_dataloader):
            
            realI = realI.to(device)
            attr_labels = attr_labels.to(device).float()
            
            noises = t.randn(realI.size()[0], 100).to(device)
            fakeI = G(t.cat((noises, attr_labels), dim=1))
            
            true = t.ones(realI.size()[0], 1).to(device)
            false = t.zeros(fakeI.size()[0], 1).to(device)
            
            lossG = criterion_real(D(fakeI)[0], true) + criterion_attr(D(fakeI)[1], attr_labels)
            
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            
            
            fakeI = fakeI.detach()
            
            lossD = criterion_real(D(realI)[0], true) + criterion_real(D(fakeI)[0], false) + criterion_attr(D(realI)[1], attr_labels)  + criterion_attr(D(fakeI)[1], attr_labels)
            
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()
            
            
            lossG_meter.update(lossG.item(), realI.size()[0])
            lossD_meter.update(lossD.item(), realI.size()[0])
            
            print(i, lossG.item(), lossD.item())
            
        t.save(G.state_dict(), os.path.join(opt.ckpts_root, 'e{}.ckpt'.format(epoch)))
        
        
        G.eval()
        
        disentangleI = G(fix_noises)
        
        G.train()
        
        writer.add_scalar('lossG', lossG_meter.avg, epoch)
        writer.add_scalar('lossD', lossD_meter.avg, epoch)
        writer.add_image('realI', tv.utils.make_grid(realI[:64], normalize=True), epoch)
        writer.add_image('fakeI', tv.utils.make_grid(fakeI[:64], normalize=True), epoch)
        writer.add_image('disentangleI', tv.utils.make_grid(disentangleI, nrow=10, normalize=True), epoch)
    

if __name__ == '__main__':
    fire.Fire()