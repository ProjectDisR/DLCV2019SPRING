import torch as t
from torch import nn

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.fc = nn.Linear(101, 768)
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.ConvTranspose2d(384, 256, 5, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 192, 6, 2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 3, 6, 2, bias=False),
            nn.Tanh()
            )
            
        return
        
    def forward(self, x):
        
        x = self.fc(x)
        x = x.view(-1, 768, 1, 1)
        
        x = self.conv(x)
        
        return x
    
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
            )
        
        self.fc_realfake = nn.Linear(512*5*5, 1)
        self.fc_attr = nn.Linear(512*5*5, 1)
            
        return
    
    def forward(self, x):
        
        x = self.conv(x)
        x = x.view(-1, 512*5*5)
        
        realfake = self.fc_realfake(x)
        attr = self.fc_attr(x)
        
        return realfake, attr


class ReverseLayerF(t.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN(nn.Module):
    
    def __init__(self):
        super(DANN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, 5, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 128, 5, 1, 1),
            nn.ReLU()
            )
        
        self.fc_domain = nn.Sequential(
            nn.Linear(128*2*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
            )
        
        self.fc_class = nn.Sequential(
            nn.Linear(128*2*2, 3072),
            nn.ReLU(),
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10)
            )
        
        return
        
    def forward(self, x, alpha):
        
        x = self.conv(x)
        x = x.view(-1, 128*2*2)
        
        domain = self.fc_domain(ReverseLayerF.apply(x, alpha))
        class_ = self.fc_class(x)
        
        return x, domain, class_
    

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, 5, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 128, 5, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        
        return
        
    def forward(self, x):
        
        x = self.conv(x)
        x = x.view(-1, 128*2*2)
        
        return x
    

class Classifier(nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.fc_class = nn.Sequential(
            nn.Linear(128*2*2, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(),
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 10)
            )
        
        return
        
    def forward(self, x):
        
        class_ = self.fc_class(x)
        
        return class_
    

class Domain_Discriminator(nn.Module):
    
    def __init__(self):
        super(Domain_Discriminator, self).__init__()
        
        self.fc_domain = nn.Sequential(
            nn.Linear(128*2*2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 1)
            )
        
        return
        
    def forward(self, x):
        
        domain = self.fc_domain(x)
        
        return domain
    
if __name__ == '__main__':
    print('Check if My_Moel works!')
    a = t.randn(2, 3, 28, 28)
    my_model = DANN()
    b, c, d = my_model(a, 0)
    print(b.size(), c.size(), d.size())
