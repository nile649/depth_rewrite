import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import data_loader
from model.network import Depth_SARPN
from options import get_args
from model.util import *


args = get_args('train')
args.root_path = os.getcwd()
print(args.root_path)
TrainImgLoader = data_loader.getTrainingData_NYUDV2(args.batch_size, args.root_path+args.trainlist_path, args.root_path+'/data/')
depth_model = Depth_SARPN(args)

# Evaluate
# TestImgLoader = data_loader.getTestingData_NYUDV2(1, args.root_path+args.trainlist_path, args.root_path+'/data/')


def train():
	print('learning rate ----> {}'.format(depth_model.get_lr()))
	for epoch in range(0,args.epochs):
		mode = "train"
		for idx, sample in enumerate(TrainImgLoader):
			depth_model.setInput(sample)
			depth_model.optimize_parameters()

			depth_model.print_loss(args.epochs,epoch,TrainImgLoader,mode)
		if (epoch+1)%args.save_itr ==0:
			depth_model.save_network(epoch)
			# depth_model.save_checkpoint(epoch)
		# evaluate(epoch)
# 
# def evaluate(epoch):
# 	for idx, sample in enumerate(TestImgLoader):
# 	        depth_model.setInput(sample)
# 	        depth_model.evaluate()
# 	depth_model.print_test()





if __name__ == '__main__':
	train()