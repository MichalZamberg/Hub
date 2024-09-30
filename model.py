from torchvision import models as models
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_chn, num_classes):
        super(Model, self).__init__()
        self.model = models.resnet18(num_classes=num_classes)
        self.softmax = nn.Softmax(dim=1)

        #Change the resent to fit the input of the network, meaning you should change the input size of the resnet from 3 channels (RGB) to input_chn.

        
    def forward(self, x, return_loss=False):
        # Complete the function, this should return the output of the resent when training.
        # When you are not training (i.e when self.training is False and when no_softmax is False) you should return the Softmax on the output of the resnet.
        # When no_softmax is True you will return the output of the network before the softmax.
    
        return x