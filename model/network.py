import os
import torch
import sys
import torch.nn as nn
from model.layers import *
from model.feature_extractor import get_models
from model.base_net import BaseNet
from model.base_model import BaseModel
from model.loss import *
from model.util import *
import pdb
import time
from model.ordinal import *


class SARPN(BaseNet): # Previously Used SARPN name
	def __init__(self,net):
	    super(SARPN, self).__init__()
	    self.feature_extraction = net
	    block_channel = [128, 256, 512,1024,2048]
	    self.residual_pyramid_decoder = ASPP_Decoder()
	    self.adaptive_dense_feature_fusion = ASPP_Encoder(block_channel)

	def forward(self, x, y):
		# pdb.set_trace()
		feature_pyramid = self.feature_extraction(x)
		fused_feature_pyramid = self.adaptive_dense_feature_fusion(feature_pyramid,y)
		multiscale_depth = self.residual_pyramid_decoder(fused_feature_pyramid)

		return multiscale_depth

# class SARPN(BaseNet): # Previously Used SARPN name
# 	def __init__(self,net):
# 	    super(SARPN, self).__init__()
# 	    self.feature_extraction = net
# 	    block_channel = [128, 256, 512,1024,2048]
# 	    self.residual_pyramid_decoder = MSW_Decoder()
# 	    self.adaptive_dense_feature_fusion = MSW_Encoder(block_channel)

# 	def forward(self, x, y):
# 		# pdb.set_trace()
# 		feature_pyramid = self.feature_extraction(x)
# 		fused_feature_pyramid = self.adaptive_dense_feature_fusion(feature_pyramid,y)
# 		multiscale_depth = self.residual_pyramid_decoder(fused_feature_pyramid)
# 		return multiscale_depth


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
		# self.loss = ordLoss() # ordinal loss
		self.temp_losses = AverageMeter()
		self.batch_time = AverageMeter()
		self.best_loss = sys.maxsize
		# Test metric
		self.totalNumber = 0
		self.criterion = nn.MSELoss().cuda()
		self.Ae = 0
		self.Pe = 0
		self.Re = 0
		self.Fe = 0

		self.errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
		            'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}




	def initVariables(self):

		self.image, self.depth, self.edge = self.input['image'], self.input['depth'], self.input['edge']
		self.image, self.depth, self.edge = self.image.cuda(), self.depth.cuda(), self.edge.cuda()  


	def forward_SARPN(self):
		self.pred_depth = self.SARPN_Net(self.image, self.edge)
		gt_depth = adjust_gt(self.depth, self.pred_depth)
		losses = list()
		for j in range(len(self.pred_depth)):
			loss_i = self.criterion(self.pred_depth[j], gt_depth[j])
			losses.append(loss_i)
		self.loss = sum(losses)
		# self.loss = total_loss(self.pred_depth, gt_depth)
		# self.loss = ordLoss()
		

	def backward_SARPN(self):
		self.loss.backward()

	def optimize_parameters(self):
		self.end = time.time()
		self.initVariables()
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

	def evaluate(self):
		self.start = time.time()
		self.initVariables()
		with torch.no_grad():
			self.forward_test()


	def print_loss(self,epoch,batch_idx,loader,mode="train"):
		if mode=='train':
			print(('Mode {mode} : Epoch: [{0}][{1}/{2}]\t'
			            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
			            'Loss {loss.val:.4f} ({loss.avg:.4f})'
			        .format(epoch, batch_idx, len(loader), batch_time=self.batch_time, loss=self.losses,mode=mode)))
		else:
			print(('Mode {mode} : Epoch: [{0}][{1}/{2}]\t'
			            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
			            'Loss {loss.val:.4f} ({loss.avg:.4f})'
			        .format(epoch, batch_idx, len(loader), batch_time=self.batch_time, loss=self.temp_losses,mode=mode)))

	def forward_test(self):
		start = time.time()
		self.pred_depth = self.SARPN_Net(self.image, self.edge)
		end = time.time()
		running_time = end - start
		output = torch.nn.functional.interpolate(self.pred_depth[0], size=[self.depth.size(2), self.depth.size(3)], mode='bilinear', align_corners=True)

		depth_edge = edge_detection(self.depth)
		output_edge = edge_detection(output)
		batchSize = self.depth.size(0)
		self.totalNumber = self.totalNumber + batchSize
		errors = evaluateError(output, self.depth)
		self.errorSum = addErrors(self.errorSum, errors, batchSize)
		self.averageError = averageErrors(self.errorSum, self.totalNumber)

		edge1_valid = (depth_edge > 1)
		edge2_valid = (output_edge > 1)

		nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
		A = nvalid / (self.depth.size(2)*self.depth.size(3))

		nvalid2 = np.sum(((edge1_valid + edge2_valid) ==2).float().data.cpu().numpy())
		P = nvalid2 / (np.sum(edge2_valid.data.cpu().numpy()))
		R = nvalid2 / (np.sum(edge1_valid.data.cpu().numpy()))

		F = (2 * P * R) / (P + R)

		self.Ae += A
		self.Pe += P
		self.Re += R
		self.Fe += F

	def print_test(self):
		Av = self.Ae / self.totalNumber
		Pv = self.Pe / self.totalNumber
		Rv = self.Re / self.totalNumber
		Fv = self.Fe / self.totalNumber
		print('PV', Pv)
		print('RV', Rv)
		print('FV', Fv)

		self.averageError['RMSE'] = np.sqrt(self.averageError['MSE'])
		print(self.averageError)
		self.totalNumber = 0

		self.Ae = 0
		self.Pe = 0
		self.Re = 0
		self.Fe = 0

		self.errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
		            'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
		self.averageError = None






if __name__ == '__main__':
	train()

