
import os
import torch
import torch.nn as nn


class E_resnet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_resnet, self).__init__()        
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_block0 = x

        x = self.maxpool(x)
        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        feature_pyramid = [x_block0, x_block1, x_block2, x_block3, x_block4]

        return feature_pyramid

class E_densenet(nn.Module):

    def __init__(self, original_model, num_features = 2208):
        super(E_densenet, self).__init__()        
        self.features = original_model.features

    def forward(self, x):
        x01 = self.features[0](x)
        x02 = self.features[1](x01)
        x03 = self.features[2](x02)
        x_block0 = x03
        x04 = self.features[3](x03)

        x_block1 = self.features[4](x04)
        x_block1 = self.features[5][0](x_block1)
        x_block1 = self.features[5][1](x_block1)
        x_block1 = self.features[5][2](x_block1)
        x_tran1 = self.features[5][3](x_block1)

        x_block2 = self.features[6](x_tran1)
        x_block2 = self.features[7][0](x_block2)
        x_block2 = self.features[7][1](x_block2)
        x_block2 = self.features[7][2](x_block2)
        x_tran2 = self.features[7][3](x_block2)

        x_block3 = self.features[8](x_tran2)
        x_block3 = self.features[9][0](x_block3)
        x_block3 = self.features[9][1](x_block3)
        x_block3 = self.features[9][2](x_block3)
        x_tran3 = self.features[9][3](x_block3)

        x_block4 = self.features[10](x_tran3)
        x_block4 = F.relu(self.features[11](x_block4))

        feature_pyramid = [x_block0, x_block1, x_block2, x_block3, x_block4]

        return feature_pyramid

class E_senet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_senet, self).__init__()        
        self.base = nn.Sequential(*list(original_model.children())[:-3])
    
    def forward(self, x):
        x_block0 = nn.Sequential(*list(self.base[0].children())[:-1])(x)      
        x0 = self.base[0](x)       
        x_block1 = self.base[1](x0)                                           
        x_block2 = self.base[2](x_block1)                                     
        x_block3 = self.base[3](x_block2)                                     
        x_block4 = self.base[4](x_block3)                                    
        feature_pyramid = [x_block0, x_block1, x_block2, x_block3, x_block4]
        return feature_pyramid