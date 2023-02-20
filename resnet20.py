import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# from wide_resnet import WideResNet
from torchvision.models.resnet import resnet50

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

svhn_mean= (0.4376821, 0.4437697, 0.47280442) 
svhn_std= (0.19803012, 0.20101562, 0.19703614) 
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

class Norm_layer(nn.Module):
    def __init__(self,mean,std) -> None:
        super(Norm_layer,self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1),requires_grad = False)

        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1),requires_grad = False)

    def forward(self,x):
        return x.sub(self.mean).div(self.std)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ClassifierModule(nn.Module):
    def __init__(self, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.linear = nn.Linear(channel, num_classes)
    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        return self.linear(x)


class ResNet20(nn.Module):
    def __init__(self,ResidualBlock = ResidualBlock, expansion = 1, kernel_size = 3, num_classes=10):
        super(ResNet20, self).__init__()
        self.net = nn.Sequential(
            nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride = 1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            ),
            self.make_layer(ResidualBlock, 16, 16 * expansion, kernel_size, stride=1),
            self.make_layer(ResidualBlock, 16 * expansion,32 * expansion, kernel_size, stride=2),
            self.make_layer(ResidualBlock, 32 * expansion,64 * expansion, kernel_size, stride=2),
            # nn.AvgPool2d(2),
            ClassifierModule(64 * expansion, num_classes)
        )

    def make_layer(self,block,inchannel,outchannel,num_block,stride):

        strides = [stride]+[1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(inchannel,outchannel,stride))
            inchannel = outchannel
        return nn.Sequential(*layers)

    def forward(self, x):
        # for i in range(len(self.net)):
        #     x = self.net[i](x)
        #     if i == 2:
        #         tamp = x
        # return x,tamp
        # print(self.net[1])
        # print(self.net[:4](x).shape)
        return self.net(x), self.net[:1](x)



