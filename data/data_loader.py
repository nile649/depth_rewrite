import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from data.data_transform import *


class NYUDV2Dataset(Dataset):
    """NYUV2D dataset."""

    def __init__(self, csv_file, root_path, transform=None):
        self.frame = pd.read_csv(csv_file, header=None,engine='python')
        self.transform = transform 
        self.root_path = root_path

    def __getitem__(self, idx):
        image_name = self.frame.ix[idx, 0]
        depth_name = self.frame.ix[idx, 1]
        root_path = self.root_path
        s_path = root_path+depth_name
        image = Image.open(root_path+image_name)
        depth = Image.open(root_path+depth_name)
        edge = Image.open(s_path[:-len(s_path.split('/')[-1])]+'hed_'+s_path.split('/')[-1])

        sample = {'image': image, 'depth': depth, 'edge':edge}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData_NYUDV2(batch_size, trainlist_path, root_path):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = NYUDV2Dataset(csv_file=trainlist_path,
                                        root_path = root_path,
                                        transform=transforms.Compose([
                                            Scale(256),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([256, 256], [256, 256],[256, 256]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=1, pin_memory=False)

    return dataloader_training


def getTestingData_NYUDV2(batch_size, testlist_path, root_path):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_testing = NYUDV2Dataset(csv_file=testlist_path,
                                        root_path=root_path,
                                        transform=transforms.Compose([
                                           Scale(256),
                                           CenterCrop([256, 256], [256, 256]),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=1, pin_memory=False)

    return dataloader_testing
    
