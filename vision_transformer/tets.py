from PIL import Image
import torch
from torch.utils.data import Dataset

import os
import math

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
from os.path import join
from copy import copy

#from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate, test,read_split_data1

class MRIDataset(Dataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self,
                 img_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 split_prefix='split',
                 **_):
        """
        Args:
            img_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.split = split
        self.transform = transform
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.split_prefix='split'
        df_path = join(img_dir, f'split.pretrained.{idx_fold}.csv')
        df = pd.read_csv(df_path)
        df = df[df['split']==split]
        self.df = df.reset_index()
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'MCI': 1, 'unlabeled': -1}
        self.label_list = [self.diagnosis_code[label] for label in self.df.diagnosis.values]
        self.size = self[0]['image'].numpy().size

    def __len__(self):
        # shuffle
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        sess_name = self.df.loc[idx, 'session_id']
        img_label = self.df.loc[idx, 'diagnosis']
        image_path = join(self.img_dir, 'subjects', img_name, sess_name, 'deeplearning_prepare_data', 'image_based', 't1_linear', img_name + '_' + sess_name + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')

        image = torch.load(image_path)
        img=img.squeeze(dim=0)
        image_shape=img.shape
        result=torch.empty( 0, 224,224)
        for dimension in range(3):
            for i in range(7):
                if dimension==0:
                    slice=img[i*image_shape[0]//7:i*image_shape[0]//7+1,:,:]
                    slice=slice.resize_(1,224,224)
                elif dimension==1:
                    slice=img[:,i*image_shape[1]//7:i*image_shape[1]//7+1,:]
                    #print(slice.shape)
                    slice=slice.permute(1,0,2)
                    slice=slice.resize_(1,224,224)
                else:
                    slice=img[:,:,i*image_shape[2]//7:i*image_shape[2]//7+1]
                    slice=slice.permute(2,0,1)
                    slice=slice.resize_(1,224,224)
                #slice=MyDataSet.change(slice)
                result=torch.cat((result, slice), 0)
        image=result
        label = self.diagnosis_code[img_label]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'participant_id': img_name, 'session_id': sess_name,
                  'image_path': image_path}

        return sample

    def session_restriction(self, session):
        """
            Allows to generate a new MRIDataset using some specific sessions only (mostly used for evaluation of test)
            :param session: (str) the session wanted. Must be 'all' or 'ses-MXX'
            :return: (DataFrame) the dataset with the wanted sessions
            """

        data_output = copy(self)
        if session == "all":
            return data_output
        else:
            df_session = self.df[self.df.session_id == session]
            df_session.reset_index(drop=True, inplace=True)
            data_output.df = df_session
            if len(data_output) == 0:
                raise Exception("The session %s doesn't exist for any of the subjects in the test data" % session)
            return data_output
        
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __change__(img):
        img=transforms.Resize([224,224])(img)
        #img_up=transforms.Resize([224,224])(img_up)
        #img_down=transforms.Resize([224,224])(img_down)
        img=transforms.RandomHorizontalFlip()(img)
        #img_up=transforms.RandomHorizontalFlip()(img_up)
        #img_down=transforms.RandomHorizontalFlip()(img_down)
        img = transforms.ToTensor()(img)
        #img_up = transforms.ToTensor()(img_up)
        #img_down = transforms.ToTensor()(img_down)
        #result=torch.cat((img_up, img, img_down), 0)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        '''if item==0:
            img_up=img
        else:
            img_up=Image.open(self.images_path[item-1])
        if item==len(self.images_class)-1:
            img_down=img
        else:
            img_down=Image.open(self.images_path[item+1])'''
        #print(item)
        # RGB为彩色图片，L为灰度图片
        '''if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))'''
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
        else:
            result=torch.empty(1, 224, 224)
            #print(img.size)
            image_shape = img.size
            #???image_shape= img.shape
            for dimension in range(3):
                for i in range(image_shape[dimension]):
                    slice=img[i,:,:]
                    slice=MyDataSet.change(slice)
                    torch.cat((result, slice), 0)
            #result=img
            #print(img.shape)
            list1=[0.5]*sum(image_shape)
            img=transforms.Normalize(mean=list1, std=list1)(result)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

    

def main():
    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data("G:/uiuc/deep-learning-for-image-processing-master/pytorch_classification/vision_transformer/flower_photos")

    '''data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              #transform=data_transform['train'],
                              transform=None)'''
    train_dataset=MRIDataset('C:/Users/user/uiuc_data/data1/ADNI_CAPS', 'train', None)
    #print(train_images_path,train_images_label)
    #print(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=0)
                                               #collate_fn=train_dataset.collate_fn)
    print(len(train_loader))
    for step, data in enumerate(train_loader):
        
        images = data["image"]
        label=data["label"]
        if step==1:
            #print(images.size,labels)
            print(images)
            print(images.shape)
            print(label)
            break

main()