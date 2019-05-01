import os
import sys

import torch as t
import torchvision as tv

from configs import GANConfig, ACGANConfig
from models import Generator

from skimage.io import imsave

device = t.device("cuda")


t.manual_seed(0)

if not os.path.isdir(sys.argv[1]):
    os.makedirs(sys.argv[1])


opt = GANConfig()

noises = t.randn(32, 101).to(device)

G = Generator()
G.load_state_dict(t.load(os.path.join(opt.ckpts_root, 'e99.ckpt')))
G = G.eval().to(device)

fakeI = G(noises)
imsave(os.path.join(sys.argv[1], 'fig1_2.jpg'), tv.utils.make_grid(fakeI, normalize=True).permute(1, 2, 0).detach().cpu().numpy())


opt = ACGANConfig()

attr = t.zeros(20, 1).to(device)
attr[0:10, 0] = 1.0

noises = t.randn(10, 100).to(device)
noises = noises.repeat(2, 1)

G = Generator()
G.load_state_dict(t.load(os.path.join(opt.ckpts_root, 'e99.ckpt')))
G = G.eval().to(device)

disentangleI = G(t.cat((noises, attr), dim=1))
imsave(os.path.join(sys.argv[1], 'fig2_2.jpg'), tv.utils.make_grid(disentangleI, nrow=10, normalize=True).permute(1, 2, 0).detach().cpu().numpy())