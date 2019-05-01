import os

import torch as t
from torch.utils.data import DataLoader
from torch import nn
import torchvision as tv

from configs import GANConfig
from datasets import Face
from models import Generator, Discriminator
from utils import AverageMeter

from tensorboardX import SummaryWriter

import fire

device = t.device("cuda")

def train(**kwargs):
    
    opt = GANConfig()
    opt.parse(kwargs)
    
    data_root = 'hw3_data/face/'
    
    train_dataset = Face(data_root)
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=True)
    
    G = Generator().to(device)
    D = Discriminator().to(device)
    
    criterion_real = nn.BCEWithLogitsLoss()
    
    optimizerG = t.optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerD = t.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    writer = SummaryWriter('runs/gan/')
    
    if not os.path.isdir(opt.ckpts_root):
        os.makedirs(opt.ckpts_root)
        
    for epoch in range(opt.n_epoch):
        
        lossG_meter = AverageMeter()
        lossD_meter = AverageMeter()
        
        for i, (realI, attr_labels) in enumerate(train_dataloader):
            
            realI = realI.to(device)
            
            noises = t.randn(realI.size()[0], 101).to(device)
            fakeI = G(noises)
            
            true = t.ones(realI.size()[0], 1).to(device)
            false = t.zeros(fakeI.size()[0], 1).to(device)
            
            lossG = criterion_real(D(fakeI)[0], true)
            
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            
            
            fakeI = fakeI.detach()
            
            lossD = criterion_real(D(realI)[0], true) + criterion_real(D(fakeI)[0], false)
            
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()
            
            
            lossG_meter.update(lossG.item(), realI.size()[0])
            lossD_meter.update(lossD.item(), realI.size()[0])
            
            print(i, lossG.item(), lossD.item())
            
        t.save(G.state_dict(), os.path.join(opt.ckpts_root, 'e{}.ckpt'.format(epoch)))
        
        
        writer.add_scalar('lossG', lossG_meter.avg, epoch)
        writer.add_scalar('lossD', lossD_meter.avg, epoch)
        writer.add_image('realI', tv.utils.make_grid(realI[:64], normalize=True), epoch)
        writer.add_image('fakeI', tv.utils.make_grid(fakeI[:64], normalize=True), epoch)


if __name__ == '__main__':
    fire.Fire()