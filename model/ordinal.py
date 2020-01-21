#https://github.com/miraiaroha/ACAN/blob/master/code/

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
#sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from math import pi, sqrt

safe_log = lambda x: torch.log(torch.clamp(x, 1e-8, 1e8))


def continuous2discrete(depth, d_min, d_max, n_c):
    mask = 1 - (depth > d_min) * (depth < d_max)
    depth = torch.round(torch.log(depth / d_min) / np.log(d_max / d_min) * (n_c - 1))
    depth[mask] = 0
    return depth

def discrete2continuous(depth, d_min, d_max, n_c):
    depth = torch.exp(depth / (n_c - 1) * np.log(d_max / d_min) + np.log(d_min))
    return depth

class BaseClassificationModel_(nn.Module):
    def __init__(self, min_depth=0, max_depth=10, num_classes=80, 
                 classifierType='OR', inferenceType='soft'):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_classes = num_classes
        self.classifierType = classifierType
        self.inferenceType = inferenceType

    def decode_ord(self, y):
        batch_size, prob, height, width = y.shape
        y = torch.reshape(y, (batch_size, prob//2, 2, height, width))
        denominator = torch.sum(torch.exp(y), 2)
        pred_score = torch.div(torch.exp(y[:, :, 1, :, :]), denominator)
        return pred_score

    def hard_cross_entropy(self, pred_score, d_min, d_max, n_c):
        pred_label = torch.argmax(pred_score, 1, keepdim=True).float()
        pred_depth = discrete2continuous(pred_label, d_min, d_max, n_c)
        return pred_depth

    def soft_cross_entropy(self, pred_score, d_min, d_max, n_c):
        pred_prob = F.softmax(pred_score, dim=1).permute((0, 2, 3, 1))
        weight = torch.arange(n_c).float().cuda()
        weight = weight * np.log(d_max / d_min) / (n_c - 1) + np.log(d_min)
        weight = weight.unsqueeze(-1)
        output = torch.exp(torch.matmul(pred_prob, weight))
        output = output.permute((0, 3, 1, 2))
        return output

    def hard_ordinal_regression(self, pred_prob, d_min, d_max, n_c):
        mask = (pred_prob > 0.5).float()
        pred_label = torch.sum(mask, 1, keepdim=True)
        #pred_label = torch.round(torch.sum(pred_prob, 1, keepdim=True))
        pred_depth = (discrete2continuous(pred_label, d_min, d_max, n_c) +
                      discrete2continuous(pred_label + 1, d_min, d_max, n_c)) / 2
        return pred_depth

    def soft_ordinal_regression(self, pred_prob, d_min, d_max, n_c):
        pred_prob_sum = torch.sum(pred_prob, 1, keepdim=True)
        Intergral = torch.floor(pred_prob_sum)
        Fraction = pred_prob_sum - Intergral
        depth_low = (discrete2continuous(Intergral, d_min, d_max, n_c) +
                     discrete2continuous(Intergral + 1, d_min, d_max, n_c)) / 2
        depth_high = (discrete2continuous(Intergral + 1, d_min, d_max, n_c) +
                      discrete2continuous(Intergral + 2, d_min, d_max, n_c)) / 2
        pred_depth = depth_low * (1 - Fraction) + depth_high * Fraction
        return pred_depth

    def inference(self, y):
        if isinstance(y, list):
            y = y[-1]
        if isinstance(y, dict):
            y = y['y']
        # mode
        # OR = Ordinal Regression
        # CE = Cross Entropy
        if self.classifierType == 'OR':
            if self.inferenceType == 'soft':
                inferenceFunc = self.soft_ordinal_regression
            else:    # hard OR
                inferenceFunc = self.hard_ordinal_regression
        else:  # 'CE'
            if self.inferenceType == 'soft': # soft CE
                inferenceFunc = self.soft_cross_entropy
            else:     # hard CE
                inferenceFunc = self.hard_cross_entropy
        pred_depth = inferenceFunc(y, self.min_depth, self.max_depth, self.num_classes)
        return pred_depth

    def forward():
        raise NotImplementedError


# def make_classifier(classifierType='OR', num_classes=80, use_inter=False, channel1=1024, channel2=2048):
#     classes = 2 * num_classes
#     interout = None
#     if use_inter:
#         interout = nn.Sequential(OrderedDict([
#             ('dropout1', nn.Dropout2d(0.2, inplace=True)),
#             ('conv1',    nn.Conv2d(channel1, channel1//2, kernel_size=3, stride=1, padding=1)),
#             ('relu',     nn.ReLU(inplace=True)),
#             ('dropout2', nn.Dropout2d(0.2, inplace=False)),
#             ('upsample', nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))]))
#     classifier = nn.Sequential(OrderedDict([
#         ('conv',     nn.Conv2d(256, 160, kernel_size=1, stride=1, padding=0))
#         ]))
#     return [interout, classifier]


class Decoder(BaseClassificationModel_):
	def __init__(self, min_depth=0, max_depth=10, num_classes=80):
	    super(Decoder, self).__init__(min_depth, max_depth, num_classes=80)
	    self.class_ = nn.Conv2d(256,160,kernel_size=1,stride=1,padding=0)

	def forward(self,fused_feature_pyramid):

	    cy1 = self.class_(fused_feature_pyramid[0])
	    y1 = self.decode_ord(cy1)
	    return [y1]




class _BaseEntropyLoss2d(nn.Module):
	def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None):
	    """
	    Parameters
	    ----------
	    ignore_index : Specifies a target value that is ignored
	                   and does not contribute to the input gradient
	    reduction : Specifies the reduction to apply to the output: 
	                'mean' | 'sum'. 'mean': elemenwise mean, 
	                'sum': class dim will be summed and batch dim will be averaged.
	    use_weight : whether to use weights of classes.
	    weight : Tensor, optional
	            a manual rescaling weight given to each class.
	            If given, has to be a Tensor of size "nclasses"
	    """
	    super(_BaseEntropyLoss2d, self).__init__()
	    self.ignore_index = ignore_index
	    self.reduction = reduction
	    self.use_weights = use_weights
	    if use_weights:
	        print("w/ class balance")
	        print(weight)
	        self.weight = torch.FloatTensor(weight).cuda()
	    else:
	        print("w/o class balance")
	        self.weight = None

	def get_entropy(self, pred, label):
	    """
	    Return
	    ------
	    entropy : shape [batch_size, h, w, c]
	    Description
	    -----------
	    Information Entropy based loss need to get the entropy according to your implementation, 
	    each element denotes the loss of a certain position and class.
	    """
	    raise NotImplementedError

	def forward(self, pred, label):
	    """
	    Parameters
	    ----------
	    pred: [batch_size, num_classes, h, w]
	    label: [batch_size, h, w]
	    """
	    assert not label.requires_grad
	    assert pred.dim() == 4
	    assert label.dim() == 3
	    assert pred.size(0) == label.size(0), "{0} vs {1} ".format(pred.size(0), label.size(0))
	    assert pred.size(2) == label.size(1), "{0} vs {1} ".format(pred.size(2), label.size(1))
	    assert pred.size(3) == label.size(2), "{0} vs {1} ".format(pred.size(3), label.size(3))

	    n, c, h, w = pred.size()
	    if self.use_weights:
	        if self.weight is None:
	            print('label size {}'.format(label.shape))
	            freq = np.zeros(c)
	            for k in range(c):
	                mask = (label[:, :, :] == k)
	                freq[k] = torch.sum(mask)
	                print('{}th frequency {}'.format(k, freq[k]))
	            weight = freq / np.sum(freq) * c
	            weight = np.median(weight) / weight
	            self.weight = torch.FloatTensor(weight).cuda()
	            print('Online class weight: {}'.format(self.weight))
	    else:
	        self.weight = 1
	    if self.ignore_index is None:
	        self.ignore_index = c + 1

	    entropy = self.get_entropy(pred, label)

	    mask = label != self.ignore_index
	    weighted_entropy = entropy * self.weight

	    if self.reduction == 'sum':
	        loss = torch.sum(weighted_entropy, -1)[mask].mean()
	    elif self.reduction == 'mean':
	        loss = torch.mean(weighted_entropy, -1)[mask].mean()
	    return loss


class OrdinalRegression2d(_BaseEntropyLoss2d):
    def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None):
        super(OrdinalRegression2d, self).__init__(ignore_index, reduction, use_weights, weight)

    def get_entropy(self, pred, label):
        n, c, h, w = pred.size()
        label = label.unsqueeze(3).long()
        pred = pred.permute(0, 2, 3, 1)
        mask10 = ((torch.arange(c)).cuda() <  label).float()
        mask01 = ((torch.arange(c)).cuda() >= label).float()
        entropy = safe_log(pred) * mask10 + safe_log(1 - pred) * mask01
        return -entropy


class LossFunc(nn.Module):
    def __init__(self, min_depth=0.72, max_depth=10):
        super(LossFunc, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.AuxiliaryLoss = OrdinalRegression2d()

    def forward(self, preds, label):
        y = preds
        dis_label = continuous2discrete(label, self.min_depth, self.max_depth, 80)
        loss = 0
        loss = self.AuxiliaryLoss(y, dis_label.squeeze(1).long())
        return loss