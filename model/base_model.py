import os
import torch
import torch.nn as nn
import pdb
# this is an abstract class of final pipeline.
class BaseModel(nn.Module):
    def __init__(self):
    	super(BaseModel,self).__init__()


    def init(self,options):
    	self.opt = options
    	self.gpu_ids = options.gpu_ids
    	self.save_dir_model = options.save_dir_model
    	self.save_dir_res = options.save_dir_res
    	self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
    	self.model_names = []

    def setInput(self, inputData):
    	self.input = inputData

    def forward(self):
    	pass

    def optimize_parameters(self):
    	pass

    def get_current_losses(self):
    	pass

    def update_learning_rate(self):
    	pass


    def evaluate(self):
        pass


    def get_lr(self):
        pass

    def save_network(self,epoch):
        # pdb.set_trace()
        for name in self.model_names:
            if isinstance(name,str):
                save_filename = "{}_Net_{}.pth".format(name,epoch)
                save_dir = "./{}/{}".format(self.save_dir_model,save_filename)

                net = getattr(self,name+'_Net')

                torch.save(net.state_dict(),save_dir)


    def save_checkpoint(self,epoch):
        # pdb.set_trace()
        for name in self.model_names:
            if isinstance(name,str):
                save_filename = "{}_checkpoint_{}.pth".format(name,epoch)
                save_dir = "./{}/{}".format(self.save_dir_model,save_filename)

                net = getattr(self,name+'_Net')
                optimizer = getattr(self,'optimizer')
                loss = getattr(self,'loss')

                # torch.save(net.state_dict(),save_dir)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, save_dir)


    def load_network(self, load_path):
        for name in self.model_names:
            if isinstance(name+'_Net', str):
                net = getattr(self, name+'_Net')
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path)
                net.load_state_dict(state_dict)

    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path)
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name+'_Net')
                optimizer = getattr(self,'optimizer')

                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                print("Epoch {} : Loss {}".format(epoch,loss))
                # self.loss = checkpoint['loss']

    # print network information
    # def print_networks(self, verbose=True):
    #     print('---------- Networks initialized -------------')
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             net = getattr(self,name)
    #             num_params = 0
    #             for param in net.parameters():
    #                 num_params += param.numel()
    #             if verbose:
    #                 print(net)
    #             print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
    #     print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad