import os
import torch
import torch.nn as nn
from model.extractor_model.senet import *
from model.extractor_model.resnet import *
from model.extractor_model.densenet import *
from model.extractor_model.extractor import *

def get_models(args):
	# path = args.extractor_path
	__models__ = {
	'SENet154': lambda :E_senet(senet154(pretrained="imagenet",path=args.root_path+args.extractor_path)),
	}   
	backbone = 'SENet154'
	return __models__[backbone]()

