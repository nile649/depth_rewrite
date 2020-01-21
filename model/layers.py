import os
import torch
import torch.nn as nn
import pdb  
import torch.nn.functional as F
from torch.nn import Parameter
from collections import OrderedDict


class ASPP(nn.Module):
    def __init__(self,num_ch, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(num_ch, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.InstanceNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(num_ch, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.InstanceNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(num_ch, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.InstanceNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(num_ch, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.InstanceNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(num_ch, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.InstanceNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(256*5, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.InstanceNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        # pdb.set_trace()
        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))

        return out






# https://github.com/miraiaroha/ACAN/

class Reshape(nn.Module):
    # input the dimensions except the batch_size dimension
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)

class PixelAttentionBlock_(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(PixelAttentionBlock_, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_key = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0)),
            ('bn',   nn.BatchNorm2d(key_channels)),
            ('relu', nn.ReLU(True))]))
        self.parameter_initialization()
        self.f_query = self.f_key

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_att(self, x):
        batch_size = x.size(0)
        query = self.f_query(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map

    def forward(self, x):
        raise NotImplementedError


class SelfAttentionBlock_(PixelAttentionBlock_):
    def __init__(self, in_channels, key_channels, value_channels, scale=1):
        super(SelfAttentionBlock_, self).__init__(in_channels, key_channels)
        self.scale = scale
        self.value_channels = value_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_value = nn.Sequential(OrderedDict([
            ('conv1',  nn.Conv2d(in_channels, value_channels, kernel_size=3, stride=1, padding=1)),
            ('relu1',  nn.ReLU(inplace=True)),
            ('conv2',  nn.Conv2d(value_channels, value_channels, kernel_size=1, stride=1)),
            ('relu2',  nn.ReLU(inplace=True))]))
        self.parameter_initialization()

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size, c, h, w = x.size()
        if self.scale > 1:
            x = self.pool(x)
        sim_map = self.forward_att(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1).permute(0, 2, 1)
        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        return [context, sim_map]




class ACN_Encoder(nn.Module):
    def __init__(self, block_channel):
        super(ACN_Encoder, self).__init__()

        self.aspp1 = ASPP(block_channel[2],160)
        self.concat = ConcatBlock()
        self.aspp2 = ASPP(224,160)
    def forward(self, feature_pyramid,edge):

        # scale3_size = [feature_pyramid[2].size(2), feature_pyramid[2].size(3)]
          
        # scale_3to3 = self.upsample_scale3to3(feature_pyramid[2], scale3_size)
       
        # scale3_mff = F.relu(self.bn_scale3(self.conv_scale3(scale_3to3)))                        

        fused_feature_pyramid = self.aspp1(feature_pyramid[2])
        concat_fused_feature_pyramid = self.concat(fused_feature_pyramid,edge)
        fused_feature_pyramid = [self.aspp2(concat_fused_feature_pyramid)]

        return fused_feature_pyramid     




# ASPP Module


class ASPP_Encoder(nn.Module):

    def __init__(self,block_channel):
        super(ASPP_Encoder, self).__init__()

        self.aspp1 = ASPP(block_channel[2],256)
        self.concat = ConcatBlock()
        self.aspp2 = ASPP(256,256)
    def forward(self, feature_pyramid,edge):                      

        fused_feature_pyramid = self.aspp1(feature_pyramid[2])
        # concat_fused_feature_pyramid = self.concat(fused_feature_pyramid)#,edge)
        fused_feature_pyramid = [self.aspp2(fused_feature_pyramid)+ fused_feature_pyramid]

        return fused_feature_pyramid         
    

class ASPP_Decoder(nn.Module):

    def __init__(self, rpd_num_features = 256):
        super(ASPP_Decoder, self).__init__()                                                              
        self.conv = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                           nn.BatchNorm2d(rpd_num_features),                                                                
                                           nn.ReLU(),                                                                                   
                                           nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  
         
        self.scale = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

    def forward(self,fused_feature_pyramid):
        # pdb.set_trace()
        scale3_size = [fused_feature_pyramid[0].size(2), fused_feature_pyramid[0].size(3)]
        scale2_size = [2*fused_feature_pyramid[0].size(2), 2*fused_feature_pyramid[0].size(3)]
        scale1_size = [4*fused_feature_pyramid[0].size(2), 4*fused_feature_pyramid[0].size(3)]

        scale3_depth = self.scale(self.conv(fused_feature_pyramid[0]))
        
        scale2 = F.interpolate(fused_feature_pyramid[0], size=scale2_size,
                                    mode='bilinear', align_corners=True)
        scale2_depth = self.scale(self.conv(scale2))

        scale1 = F.interpolate(fused_feature_pyramid[0], size=scale1_size,
                                    mode='bilinear', align_corners=True)
        scale1_depth = self.scale(self.conv(scale1))

        scale_depth = [scale3_depth,scale2_depth,scale1_depth]

        return scale_depth


# class ASPP_Classifier(nn.Module):

#     def __init__(self, channel2=160,classes = 80):
#         super(ASPP_Classifier, self).__init__()

#         self.classifier = nn.Sequential(OrderedDict([
#             ('conv',     nn.Conv2d(channel2, 2*classes, kernel_size=1, stride=1, padding=0)),
#             ('upsample', nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))]))

#     def forward(self,fused_feature_pyramid):
        

#         return classi
# #     # SARPN

class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)      
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,          
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,             
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear',align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out


class Refineblock(nn.Module):
    def __init__(self, num_features, kernel_size):
        super(Refineblock, self).__init__()
        padding=(kernel_size-1)//2

        self.conv1 = nn.Conv2d(1, num_features//2, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(num_features//2)

        self.conv2 = nn.Conv2d(  
            num_features//2, num_features//2, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features//2)

        self.conv3 = nn.Conv2d(num_features//2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=True)



    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.bn1(x_res)
        x_res = F.relu(x_res)
        x_res = self.conv2(x_res)
        x_res = self.bn2(x_res)
        x_res = F.relu(x_res)
        x_res = self.conv3(x_res)

        x2 = x  + x_res
        return x2




class Decoder(nn.Module):

    def __init__(self, rpd_num_features = 2048):
        super(Decoder, self).__init__()


        self.conv = nn.Conv2d(rpd_num_features // 2, rpd_num_features // 2, kernel_size=1, stride=1, bias=False)                                               
        self.bn = nn.BatchNorm2d(rpd_num_features//2)                                                    

        self.conv5 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features//2, kernel_size=3, stride=1, padding=1, bias=False),    
                                   nn.BatchNorm2d(rpd_num_features//2),                                                             
                                   nn.ReLU(),                                                                                   
                                   nn.Conv2d(rpd_num_features//2, 1, kernel_size=3, stride=1, padding=1, bias=False))               
        rpd_num_features = rpd_num_features // 2                                                                                              
        self.scale5 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2        

        self.conv4 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features),                                                                
                                   nn.ReLU(),                                                                                   
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale4 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                        
        

        self.conv3 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features),
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale3 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                        

        self.conv2 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(rpd_num_features),
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.scale2 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                        

        self.conv1 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features),                                                                
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale1 = Refineblock(num_features=rpd_num_features, kernel_size=3)
    def forward(self,fused_feature_pyramid):

        scale1_size = [fused_feature_pyramid[0].size(2), fused_feature_pyramid[0].size(3)]
        scale2_size = [fused_feature_pyramid[1].size(2), fused_feature_pyramid[1].size(3)]
        scale3_size = [fused_feature_pyramid[2].size(2), fused_feature_pyramid[2].size(3)]
        scale4_size = [fused_feature_pyramid[3].size(2), fused_feature_pyramid[3].size(3)]
        scale5_size = [fused_feature_pyramid[4].size(2), fused_feature_pyramid[4].size(3)]
        

        # scale5
        scale5 = torch.cat((F.relu(self.bn(self.conv(fused_feature_pyramid[4]))), fused_feature_pyramid[4]), 1)
        scale5_depth = self.scale5(self.conv5(scale5))

        # scale4
        scale4_res = self.conv4(fused_feature_pyramid[3])
        scale5_upx2 = F.interpolate(scale5_depth, size=scale4_size,
                                    mode='bilinear', align_corners=True)
        scale4_depth = self.scale4(scale4_res + scale5_upx2)

        # scale3 
        scale3_res = self.conv3(fused_feature_pyramid[2])
        scale4_upx2 = F.interpolate(scale4_depth, size=scale3_size,
                                    mode='bilinear', align_corners=True)
        scale3_depth = self.scale3(scale3_res + scale4_upx2)

        # scale2
        scale2_res = self.conv2(fused_feature_pyramid[1])
        scale3_upx2 = F.interpolate(scale3_depth, size=scale2_size,
                                    mode='bilinear', align_corners=True)
        scale2_depth = self.scale2(scale2_res + scale3_upx2)

        # scale1
        scale1_res = self.conv1(fused_feature_pyramid[0])
        scale2_upx2 = F.interpolate(scale2_depth, size=scale1_size,
                                    mode='bilinear', align_corners=True)
        scale1_depth = self.scale1(scale1_res + scale2_upx2)

        scale_depth = [scale5_depth, scale4_depth, scale3_depth, scale2_depth, scale1_depth]

        return scale_depth



class Encoder(nn.Module):

    def __init__(self, block_channel, adff_num_features=1280, rpd_num_features=1280):
        super(Encoder, self).__init__()

        rpd_num_features = rpd_num_features // 2  # 640
        print("block_channel:", block_channel)
        # 2048 -> 256
        self.upsample_scale5to5 = _UpProjection(num_input_features=block_channel[4], num_output_features=adff_num_features//5)
        sum_ = adff_num_features//5
        # 256 -> 1024
        self.conv_scale5 = nn.Conv2d(sum_, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)       # 1280/1024
        self.bn_scale5 = nn.BatchNorm2d(rpd_num_features)

        adff_num_features = adff_num_features // 2     # 640           
        rpd_num_features = rpd_num_features // 2   # 


        # scale4
 
        self.upsample_scale4to4 = _UpProjection(num_input_features=block_channel[3], num_output_features=adff_num_features//5) 
        
        sum_ = adff_num_features//5
        self.conv_scale4 = nn.Conv2d(sum_, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)       # 640/512
        self.bn_scale4 = nn.BatchNorm2d(rpd_num_features) 

        adff_num_features = adff_num_features // 2   # 320              
        rpd_num_features = rpd_num_features // 2     # 

        # scale3
        
        self.upsample_scale3to3 = _UpProjection(num_input_features=block_channel[2], num_output_features=adff_num_features//5)  
        
        sum_ = adff_num_features//5  
        self.conv_scale3 = nn.Conv2d(sum_, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)       # 320/256
        self.bn_scale3 = nn.BatchNorm2d(rpd_num_features)

        adff_num_features = adff_num_features // 2     # 160             
        rpd_num_features = rpd_num_features // 2        

        # scale2
        
        self.upsample_scale2to2 = _UpProjection(num_input_features=block_channel[1], num_output_features=adff_num_features//5)  
         
        sum_ = adff_num_features//5   
        self.conv_scale2 = nn.Conv2d(sum_, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)        # 160/128
        self.bn_scale2 = nn.BatchNorm2d(rpd_num_features)

        adff_num_features = adff_num_features // 2   # 80               
        rpd_num_features = rpd_num_features // 2     

        #scale1   
        self.upsample_scale1to1 = _UpProjection(num_input_features=block_channel[0], num_output_features=adff_num_features//5)  
     
        sum_ = adff_num_features//5 
        self.conv_scale1 = nn.Conv2d(sum_, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False)        # 80/64
        self.bn_scale1 = nn.BatchNorm2d(rpd_num_features)
    
    def forward(self, feature_pyramid):
        scale1_size = [feature_pyramid[0].size(2), feature_pyramid[0].size(3)]
        scale2_size = [feature_pyramid[1].size(2), feature_pyramid[1].size(3)]
        scale3_size = [feature_pyramid[2].size(2), feature_pyramid[2].size(3)]
        scale4_size = [feature_pyramid[3].size(2), feature_pyramid[3].size(3)]
        scale5_size = [feature_pyramid[4].size(2), feature_pyramid[4].size(3)]

        # pdb.set_trace()
      
        scale_5to5 = self.upsample_scale5to5(feature_pyramid[4], scale5_size)
        
        scale5_mff = F.relu(self.bn_scale5(self.conv_scale5(scale_5to5)))                         

        # scale4_mff       15x19
     
        scale_4to4 = self.upsample_scale4to4(feature_pyramid[3], scale4_size)
        
        scale4_mff = F.relu(self.bn_scale4(self.conv_scale4(scale_4to4)))                        

        # scale3_mff       29x38
      
        scale_3to3 = self.upsample_scale3to3(feature_pyramid[2], scale3_size)
       
        scale3_mff = F.relu(self.bn_scale3(self.conv_scale3(scale_3to3)))                        

        # scale2_mff      57x76
       
        scale_2to2 = self.upsample_scale2to2(feature_pyramid[1], scale2_size)
       
        scale2_mff = F.relu(self.bn_scale2(self.conv_scale2(scale_2to2)))                        

        # scale1_mff      114x152
        scale_1to1 = self.upsample_scale1to1(feature_pyramid[0], scale1_size)             
        scale1_mff = F.relu(self.bn_scale1(self.conv_scale1(scale_1to1)))                           

        fused_feature_pyramid = [scale1_mff, scale2_mff, scale3_mff, scale4_mff, scale5_mff]

        return fused_feature_pyramid    

    
class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 3),
                        nn.InstanceNorm2d(out_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class ConcatBlock(nn.Module):

    def __init__(self, input_nc=1, in_features=160, out_features=224):
        super(ConcatBlock, self).__init__()
        self.channels = in_features

        self.convblock = ConvBlock(in_features + in_features, out_features)
        self.up_conv = nn.Conv2d(in_features * 2, in_features, 3, 1, 1)
        self.down_conv = nn.Sequential(
            nn.Conv2d(64, in_features // 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_features // 4, in_features // 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_features // 2, in_features, 1, 1),
            nn.ReLU()
        )
        # self.noise = NoiseInjection(in_features)

        self.convblock_ = ConvBlock(in_features + 64, out_features)

        self.vgg_block = nn.Sequential(
            nn.Conv2d(input_nc, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1, 1),
            nn.ReLU()
        )

    def forward(self, x, edge=None):
        
            edge = F.upsample(edge, size=(x.shape[2], x.shape[2]), mode='bilinear')

            edge = self.vgg_block(edge)
            concat = torch.cat([x, edge], 1)

            out = (self.convblock_(concat))
            return out



# Wasp Module - Bruno Artacho : Email : bmartacho@mail.rit.edu 
class _AtrousModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, BatchNorm):
        super(_AtrousModule, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# Residual Pyramid Decoder
class WASP(nn.Module):
    def __init__(self, in_channels):
        super(WASP, self).__init__()
        # inplanes = top_num_features
        dilations = [24, 18, 12,  6]

        self.wasp1 = _AtrousModule(in_channels, 256, 1, padding=0, dilation=dilations[0], BatchNorm=nn.BatchNorm2d)
        self.wasp2 = _AtrousModule(256, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=nn.BatchNorm2d)
        self.wasp3 = _AtrousModule(256, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=nn.BatchNorm2d)
        self.wasp4 = _AtrousModule(256, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=nn.BatchNorm2d)

        self.global_avg_pool = nn.Sequential(nn.Conv2d(in_channels, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1   = nn.Conv2d(1280, 256, 1, bias=False)
        self.conv2   = nn.Conv2d(256,256,1,bias=False)
        self.bn1     = nn.BatchNorm2d(256)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # self._init_weight()


    def forward(self, x):
        x1 = self.wasp1(x)
        x2 = self.wasp2(x1)
        x3 = self.wasp3(x2)
        x4 = self.wasp4(x3)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)
        
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)





# Residual Pyramid Decoder
class MSW_Decoder(nn.Module):

    def __init__(self, rpd_num_features = 256):
        super(MSW_Decoder, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features//2, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features//2),                                                                
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features//2, rpd_num_features//4, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features//4),                                                                
                                   nn.ReLU(), 
                                   nn.Conv2d(rpd_num_features//4, rpd_num_features//8, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features//8),                                                                
                                   nn.ReLU(),                                   
                                   nn.Conv2d(rpd_num_features//8, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale1 = Refineblock(num_features=rpd_num_features//8, kernel_size=3)

    def forward(self,fused_feature_pyramid):

        scale1_size = [fused_feature_pyramid[0].size(2), fused_feature_pyramid[0].size(3)]
        scale2_size = [fused_feature_pyramid[1].size(2), fused_feature_pyramid[1].size(3)]
        scale3_size = [fused_feature_pyramid[2].size(2), fused_feature_pyramid[2].size(3)]
        scale4_size = [fused_feature_pyramid[3].size(2), fused_feature_pyramid[3].size(3)]
        scale5_size = [fused_feature_pyramid[4].size(2), fused_feature_pyramid[4].size(3)]
        

        # scale5
        scale5_depth = self.scale1(self.conv1(fused_feature_pyramid[4]))

        # scale4
        scale4_res = self.conv1(fused_feature_pyramid[3])
        scale5_upx2 = F.interpolate(scale5_depth, size=scale4_size,
                                    mode='bilinear', align_corners=True)
        scale4_depth = self.scale1(scale4_res + scale5_upx2)

        # scale3 
        scale3_res = self.conv1(fused_feature_pyramid[2])
        scale4_upx2 = F.interpolate(scale4_depth, size=scale3_size,
                                    mode='bilinear', align_corners=True)
        scale3_depth = self.scale1(scale3_res + scale4_upx2)

        # scale2
        scale2_res = self.conv1(fused_feature_pyramid[1])
        scale3_upx2 = F.interpolate(scale3_depth, size=scale2_size,
                                    mode='bilinear', align_corners=True)
        scale2_depth = self.scale1(scale2_res + scale3_upx2)

        # scale1
        scale1_res = self.conv1(fused_feature_pyramid[0])
        scale2_upx2 = F.interpolate(scale2_depth, size=scale1_size,
                                    mode='bilinear', align_corners=True)
        scale1_depth = self.scale1(scale1_res + scale2_upx2)

        scale_depth = [scale5_depth, scale4_depth, scale3_depth, scale2_depth, scale1_depth]

        return scale_depth




class MSW_Encoder(nn.Module):
    def __init__(self,block_channel):
        super(MSW_Encoder, self).__init__()

        self.wasp0   = WASP(128)  
        self.wasp1   = WASP(256)
        self.wasp2   = WASP(512)
        self.wasp3   = WASP(1024)
        self.wasp4   = WASP(2048)
        self.concat0 = ConcatBlock(1,256,320)
        self.waspy   = WASP(320)

    def forward(self, feature_pyramid, y):
        # pdb.set_trace()
################# 256 is o/p of wasp0 ###################
        x0 = self.wasp0(feature_pyramid[0])
        x1 = self.wasp1(feature_pyramid[1])
        x2 = self.wasp2(feature_pyramid[2])
        x3 = self.wasp3(feature_pyramid[3])
        x4 = self.wasp4(feature_pyramid[4])

################# 320 is o/p of wasp0 ###################
        c0 = self.concat0(x0,y)
        c1 = self.concat0(x1,y)
        c2 = self.concat0(x2,y)
        c3 = self.concat0(x3,y)
        c4 = self.concat0(x4,y)

################# 256 is o/p of wasp0 ###################
        y0 = self.waspy(c0)
        y1 = self.waspy(c1)
        y2 = self.waspy(c2)
        y3 = self.waspy(c3)
        y4 = self.waspy(c4)


        fused_feature_pyramid = [y0, y1, y2, y3, y4]

        return fused_feature_pyramid    
