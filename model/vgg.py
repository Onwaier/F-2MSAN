'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 7) # 针对输入为48 * 48
        # self.classifier = nn.Linear(8192, 8) # 针对输入为128 * 128
        self.classifier = nn.Linear(8192, 7) # 针对输入为128 * 128

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG1(nn.Module):
    def __init__(self, vgg_name):
        super(VGG1, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 7) # 针对输入为48 * 48
        self.classifier = nn.Linear(8192, 8) # 针对输入为128 * 128

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        # out = F.dropout(out, p=0.5, training=self.training)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_reg(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_reg, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 7) # 针对输入为48 * 48
        self.hidden1 = nn.Linear(8, 16)
        self.hidden2 = nn.Linear(16, 32)
        self.classifier = nn.Linear(32, 1) # 针对输入为128 * 128

    def forward(self, x):
        # out = self.features(x)
        # out = out.view(out.size(0), -1)
        # out = F.dropout(out, p=0.5, training=self.training)
        out = F.relu(self.hidden1(x))
        out = F.relu(self.hidden2(out))
        out = self.classifier(out)
        # out = F.dropout(out, p=0.5, training=self.training)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == "__main__":
    net = VGG('VGG19')
    # print(net)
    input = torch.randn(1, 3, 128, 128)
    # net(input)
    print(net(input).size())