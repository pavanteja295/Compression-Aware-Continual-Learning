# setting bias to false for now
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import init

class Net(nn.Module):
    """Small architechture"""
    def __init__(self,num_classes=10):
        super(Net, self).__init__()
        # self.act=OrderedDict()
        self.conv1 = nn.Conv2d(3, 64, 3, bias=True)
        self.conv2 = nn.Conv2d(64, 64, 3, bias=True)
        self.drop_outA = nn.Dropout(0.15)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=True)
        self.conv4 = nn.Conv2d(128,128,3, bias=True)
        self.drop_outB = nn.Dropout(0.15)
        self.conv5 = nn.Conv2d(128,256,2, bias=True)
        self.last = nn.Linear(256*8*8, num_classes)
        #self.last = nn.Linear(256*4, num_classes)

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop_outA(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop_outB(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, 2)
        # x = x.view(-1, 256*4)
        x = x.view(-1, 256*8*8)
        x = self.logits(x)
        return x
