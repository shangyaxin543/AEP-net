import os
import numpy as np
import torch.nn as nn
import torch

class URIM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.lc_conv_5 = nn.Conv2d(in_channels = 192, out_channels = 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, initial_maps, seg_features):
        initial_maps1 = (initial_maps/torch.max(initial_maps))*10
        score = torch.sigmoid(initial_maps1)
        dist = torch.abs(score-0.75)
        confidence_map = (0.5 - (dist / 0.5))

        
        
        # print('seg_features:',seg_features)/ (F.avg_pool2d(confidence_map, 3, 1, padding=1) * 9)
        r = self.conv(seg_features * confidence_map+seg_features) 
        r = self.relu(r)
       

        r = torch.cat([r, seg_features], dim=1)
        r = self.lc_conv_5(r)
        r = self.relu(r)
        # r = self.classifier(r)
        return r, confidence_map

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_x = nn.Conv2d(out_ch, out_ch, kernel_size=(5,3), stride=1, padding=(2,1))
        self.conv_y = nn.Conv2d(out_ch, out_ch, kernel_size=(3,5), stride=1, padding=(1,2))
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_t = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_t_last = nn.ConvTranspose2d(out_ch, 1, kernel_size=3, stride=1, padding=1)
       # self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        #在网络中加入不确定区域模块
        self.urim = URIM()

    def forward(self, x):
        # encoder
        residual_1 = x.clone()
        
        out = self.relu(self.conv_first(x))
        out_x = self.relu(self.conv_x(out))
        out_y = self.relu(self.conv_y(out))
        out = out_x + out_y
        residual_2 = out.clone()
        out_x = self.relu(self.conv_x(out))
        out_y = self.relu(self.conv_y(out))
        out = out_x + out_y
        residual_3 = out.clone()
        out_x = self.relu(self.conv_x(out))
        out_y = self.relu(self.conv_y(out))
        out = out_x + out_y
        residual_4 = out.clone()
        out_x = self.relu(self.conv_x(out))
        out_y = self.relu(self.conv_y(out))
        out = out_x + out_y 
        residual_5 = out.clone()
        out_x = self.relu(self.conv_x(out))
        out_y = self.relu(self.conv_y(out))
        out = out_x + out_y

        out = self.conv_t(self.relu(out))
        out += residual_5
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))
        out += residual_4
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))
        out += residual_3
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))
        out += residual_2
        out1 = self.conv_t(self.relu(out))

        out = self.conv_t_last(self.relu(out1))
        out += residual_1

        out = self.relu(out)

        x, initial_seg = self.urim(out, out1)
        


        return x,  out
