import torch
import numpy as np
import io
import os
import os.path as osp
import importlib
import random
import time
import csv
from collections import namedtuple
import lmdb
from tqdm import tqdm
import pickle
import cv2
import yaml
import uuid
import glob
from easydict import EasyDict as edict
from torch.utils.data import Dataset
import sys
from PIL import Image
from io import BytesIO
import taming.data.talking_face.audio.AudioConfig as audio
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torchvision import datasets, transforms
import albumentations
import albumentations.pytorch.transforms
from taming.data.talking_face.lmdb_utils import LMDBReader
from utils.import_utils import instantiate_from_config
from skimage.io import imread, imsave


class ObamaDataset(Dataset):
    def __init__(self, data_config_path):
        with open(data_config_path, 'r') as f:
            args = yaml.load(f,Loader=yaml.FullLoader)
            args = edict(args)
            args = args.obama

        self.opt = args
        
        self.images_root = os.path.join(self.opt.data_root, 'images')

        if self.opt.isTrain:
            self.images_dir = os.path.join(self.images_root, 'train')
            self.images = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))[:]
        else:
            self.images_dir = os.path.join(self.images_root, 'test')
            self.images = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))[:800]

        self.len_file = len(self.images) // self.opt.batch_internal
        print('Loading data from %s, total number of images is %d'%(self.images_dir, self.len_file)) 


    def __getitem__(self, index):
        
        batch_img = []
        
        for i in range(self.opt.batch_internal):
            idx = index*self.opt.batch_internal + i
            image = self.get_image(self.images[idx])
            batch_img.append(image)

        self.input_image = torch.cat(batch_img, 0)

        data_dict = {'image': self.input_image}

        return data_dict

    def get_image(self, path):
        image = imread(path)
        image = albumentations.pytorch.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})(image=image)['image']
        return image

    def __len__(self):
        return len(self.images) // self.opt.batch_internal