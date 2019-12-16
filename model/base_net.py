import os
import torch
import torch.nn as nn



class BaseNet(nn.Module):
    def __init__(self):
    	super(BaseNet,self).__init__()

    def init(self,options):
    	self.opt = options
    	self.gpu_ids = options.gpu_ids
    	self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

    def forward(self, *input):
        return super(BaseNet, self).forward(*input)

    def test(self, *input):
        with torch.no_grad():
            self.forward(*input)

    def feature_extractor(self,model):
    	pass
