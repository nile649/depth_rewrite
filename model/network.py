import os
import torch
import torch.nn as nn
from model.layers import *
from model.feature_extractor import get_models
from model.base_net import BaseNet
from model.base_model import BaseModel
from model.loss import *
from model.util import *
import pdb
import time


class SARPN(BaseNet):
    def __init__(self,net):
        super(SARPN, self).__init__()
        self.feature_extraction = net
        adff_num_features = 1280
        rpd_num_features = 1280
        block_channel = [128, 256, 512,1024,2048]
        top_num_features = block_channel[-1]
        self.residual_pyramid_decoder = Decoder(rpd_num_features)
        self.adaptive_dense_feature_fusion = Encoder(block_channel, adff_num_features, rpd_num_features)

    def forward(self, x):
        feature_pyramid = self.feature_extraction(x)
        fused_feature_pyramid = self.adaptive_dense_feature_fusion(feature_pyramid)
        multiscale_depth = self.residual_pyramid_decoder(fused_feature_pyramid)

        return multiscale_depth


class Depth_SARPN(BaseModel):
	def __init__(self,args):
		super(Depth_SARPN,self).__init__()
		Enet = get_models(args)
		self.SARPN_Net = SARPN(Enet).cuda()
		self.opt = args
		self.init(args)
		self.optimizer = build_optimizer(model = self.SARPN_Net,
                learning_rate=args.lr,
                optimizer_name=args.optimizer,
                weight_decay = 1e-5,
                epsilon=0.001,
                momentum=0.9
                )
		self.model_names = ['SARPN']
		self.losses = AverageMeter()
		self.batch_time = AverageMeter()

	def initVariables(self):

		self.image, self.depth = self.input['image'], self.input['depth']
		self.image, self.depth = self.image.cuda(), self.depth.cuda()  


	def forward_SARPN(self):
		gt_depth = adjust_gt(self.depth, self.pred_depth)
		self.loss = total_loss(self.pred_depth, gt_depth)
		

	def backward_SARPN(self):
		self.loss.backward()

	def optimize_parameters(self):
		self.end = time.time()
		self.initVariables()
		self.pred_depth = self.SARPN_Net(self.image)
		self.optimizer.zero_grad()
		self.forward_SARPN()
		self.losses.update(self.loss.item(), self.image.size(0))
		self.backward_SARPN()
		self.optimizer.step()
		self.batch_time.update(time.time() - self.end)
		self.end = time.time()

	def get_current_loss(self):
		return self.losses.val

	def get_lr(self):
	    for param_group in self.optimizer.param_groups:
	        return param_group['lr']

	def print_loss(self,epoch,batch_idx,loader):
		print(('Epoch: [{0}][{1}/{2}]\t'
		            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
		            'Loss {loss.val:.4f} ({loss.avg:.4f})'
		        .format(epoch, batch_idx, len(loader), batch_time=self.batch_time, loss=self.losses)))


