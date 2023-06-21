import sys
sys.path.append('../')

import os
import math
import argparse
import pprint
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
#from tensorboardX import SummaryWriter

from dataManagement.dataset import get_dataloader
from dataManagement.transform import get_transform, MinMaxNormalization
from utility import utils
from utility import config_utils
from utility import checkpoint_utils

def run(config):
    dataloaders = {}
    dataloaders['train'] = get_dataloader(config, 'train', weighted_sampler=False)
    dataloaders['val'] = get_dataloader(config, 'val', weighted_sampler=False)
    tbar = tqdm.tqdm(enumerate(dataloaders), total=2)
    for i, data in tbar:
        images = data['image']
        # pgd_images = pgd_attack(data['image'], data['label']).cuda()
        labels = data['label']
        print(images,labels)
def parse_args():
    parser = argparse.ArgumentParser(description='ADNI')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default="C:/Users/user/uiuc_data/data1/ADNI_CAPS/", type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('train ADNI1/ADNIGO/ADNI2 1.5T MRI images.')
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = config_utils.load(args.config_file)
    #pprint.PrettyPrinter(indent=2).pprint(config)
    #utils.prepare_train_directories(config)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()