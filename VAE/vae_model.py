# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:51:51 2024

@author: OMEN
"""
import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm 
from torchvision.utils import save_image, make_grid 


class ConvEncoder(nn.Module):
    """
    """
    
    def __init__(self, input_channels, hidden_dim, latent_dim):
        super(ConvEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, hidden_dim, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2,2) 
        
        # last layers
        
        # for mean of the latent variable
        self.conv_mean = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        
        # for standard deviation of latent variable
        self.conv_sigma = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.LeakyReLU(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        mean = self.avgpool(self.conv_mean(x))
        sigma = self.avgpool(self.conv_sigma(x))
        
        print(mean.shape, sigma.shape)
        
        mean = mean.squeeze(-1).squeeze(-1)
        sigma = sigma.squeeze(-1).squeeze(-1)
        
        return mean, sigma 