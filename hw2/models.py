import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['vgg16_bn']
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, features, output_size=1274, image_size=448):
        super(VGG, self).__init__()
        self.features = features
        self.image_size = image_size

        self.yolo = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(4096, output_size)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.yolo(x)
        x = torch.sigmoid(x) 
        x = x.view(-1, 26, 7, 7)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    s = 1
    first_flag=True
    for v in cfg:
        s=1
        if (v==64 and first_flag):
            s=2
            first_flag=False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def conv_bn_relu(in_channels,out_channels,kernel_size=3,stride=2,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}




def Yolov1_vgg16bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    yolo = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        yolo_state_dict = yolo.state_dict()
        for k in vgg_state_dict.keys():
            if k in yolo_state_dict.keys() and k.startswith('features'):
                yolo_state_dict[k] = vgg_state_dict[k]
    yolo.load_state_dict(yolo_state_dict)
    return yolo

#class Bottleneck(nn.Module):
#    expansion = 4
#
#    def __init__(self, inplanes, planes, stride=1, downsample=None):
#        super(Bottleneck, self).__init__()
#        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                               padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(planes)
#        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#        self.bn3 = nn.BatchNorm2d(planes * 4)
#        self.relu = nn.ReLU(inplace=True)
#        self.downsample = downsample
#        self.stride = stride
#
#    def forward(self, x):
#        residual = x
#
#        out = self.conv1(x)
#        out = self.bn1(out)
#        out = self.relu(out)
#
#        out = self.conv2(out)
#        out = self.bn2(out)
#        out = self.relu(out)
#
#        out = self.conv3(out)
#        out = self.bn3(out)
#
#        if self.downsample is not None:
#            residual = self.downsample(x)
#
#        out += residual
#        out = self.relu(out)
#
#        return out
#
#class detnet_bottleneck(nn.Module):
#    # no expansion
#    # dilation = 2
#    # type B use 1x1 conv
#    expansion = 1
#
#    def __init__(self, in_planes, planes, stride=1, block_type='A'):
#        super(detnet_bottleneck, self).__init__()
#        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
#        self.bn2 = nn.BatchNorm2d(planes)
#        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#        self.downsample = nn.Sequential()
#        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
#            self.downsample = nn.Sequential(
#                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(self.expansion*planes)
#            )
#
#    def forward(self, x):
#        out = F.relu(self.bn1(self.conv1(x)))
#        out = F.relu(self.bn2(self.conv2(out)))
#        out = self.bn3(self.conv3(out))
#        out += self.downsample(x)
#        out = F.relu(out)
#        return out
#
#
#class ResNet(nn.Module):
#
#    def __init__(self, block, layers):
#        self.inplanes = 64
#        super(ResNet, self).__init__()
#        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                               bias=False)
#        self.bn1 = nn.BatchNorm2d(64)
#        self.relu = nn.ReLU(inplace=True)
#        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#        self.layer1 = self._make_layer(block, 64, layers[0])
#        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#        # self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
#        self.layer5 = self._make_detnet_layer(in_channels=2048)
#        # self.avgpool = nn.AvgPool2d(14) #fit 448 input size
#        # self.fc = nn.Linear(512 * block.expansion, num_classes)
#        self.conv_end = nn.Conv2d(256, 26, kernel_size=3, stride=2, padding=1, bias=False)
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()
#
#    def _make_layer(self, block, planes, blocks, stride=1):
#        downsample = None
#        if stride != 1 or self.inplanes != planes * block.expansion:
#            downsample = nn.Sequential(
#                nn.Conv2d(self.inplanes, planes * block.expansion,
#                          kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(planes * block.expansion),
#            )
#
#        layers = []
#        layers.append(block(self.inplanes, planes, stride, downsample))
#        self.inplanes = planes * block.expansion
#        for i in range(1, blocks):
#            layers.append(block(self.inplanes, planes))
#
#        return nn.Sequential(*layers)
#    
#    def _make_detnet_layer(self,in_channels):
#        layers = []
#        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type='B'))
#        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
#        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        x = self.maxpool(x)
#
#        x = self.layer1(x)
#        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)
#        x = self.layer5(x)
#        # x = self.avgpool(x)
#        # x = x.view(x.size(0), -1)
#        # x = self.fc(x)
#        x = self.conv_end(x)
#        x = torch.sigmoid(x)
#
#        return x
#
#def Yolov1_resnet(pretrained=False, **kwargs):
#    """Constructs a ResNet-50 model.
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#    return model

def test():
    import torch
    model = Yolov1_vgg16bn(pretrained=True)
    img = torch.rand(2, 3, 448, 448)
    output = model(img)
    print(output.size())

if __name__ == '__main__':
    test()
