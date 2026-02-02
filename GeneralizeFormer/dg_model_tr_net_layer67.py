import os
import sys
import time
import math
import random
import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import models
import numpy as np 
import pdb
import torch.nn.functional as f
# from main import args

resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)


#cretae new class for resnet50
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.features = resnet50
        self.fc = nn.Linear(1000, num_classes)
        # self.init_params()
    def forward(self, x, ctx):
        z0 = self.features(x)
        #z1 = z0.view(z0.size(0), -1)
        z2 = self.fc(z0)
        #now pass ctx 
        ctx = self.features(x)
        #ctx = ctx.view(ctx.size(0), -1)
        ctx = self.fc(ctx)
        return ctx, z2


import transformer_model_confg_67

class ResNet18_tr_net(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(ResNet18_tr_net, self).__init__()
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        # self.features = resnet18
        

        self.fc = nn.Linear(512, num_classes)
        self.tr_net = transformer_model_confg_67.TransformerModel_67(feature_dim=512, num_class= 8)
    def forward(self, x,):

        x = self.features(x)
       
        x_f = x.view(x.size(0), -1)
        
        
        
        
        x = self.fc(x_f)
        # #now pass ctx
        return x, x_f
    