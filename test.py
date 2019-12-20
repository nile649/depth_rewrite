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


args = get_args('test')
args.root_path = os.getcwd()
print(args.root_path)
depth_model = Depth_SARPN(args)

# Evaluate
TestImgLoader = data_loader.getTestingData_NYUDV2(1, args.root_path+args.testlist_path, args.root_path+'/data/')


def test():
	depth_model.load_network("./SARPN_Net_5.pth")
	for idx, sample in enumerate(TestImgLoader):
	    depth_model.setInput(sample)
	    depth_model.evaluate()
	depth_model.print_test()




if __name__ == '__main__':
	test()