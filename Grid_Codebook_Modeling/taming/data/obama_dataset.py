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
        self.ldmks_root = os.path.join(self.opt.data_root, 'landmarks')

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
        batch_ldmk = []

        if self.opt.isTrain:
            scale = random.uniform(0.95, 1.05)
        else:
            scale = 1.0
        
        for i in range(self.opt.batch_internal):
            idx = index*self.opt.batch_internal + i
            image = self.get_image(self.images[idx])
            batch_img.append(image)

        self.input_image = torch.cat(batch_img, 0)

        data_dict = {'images': self.input_image}


        return data_dict

    def get_image(self, path, scale=1.0):
        if 'jpg' in path:
            image = imread(path)
            image = albumentations.pytorch.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})(image=image)['image']
        elif 'txt' in path:
            ldmk = np.loadtxt(path)
            ldmk = self.get_scale(ldmk, scale)
            image = self.draw_heatmap_from_68_landmark(ldmk)
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image.astype(np.float32)/255)
        return image

    # def get_ldmk(self, path):
    #     ldmk = np.loadtxt(path).astype(np.float32)
    #     ldmk = np.expand_dims(ldmk, 0)
    #     ldmk = torch.from_numpy(ldmk)
    #     return ldmk

    def __len__(self):
        return len(self.images) // self.opt.batch_internal

    def get_scale(self, ldmk, scale=0.95):
        center = self.get_center(ldmk)
        offset = ldmk - center
        new_ldmk = center + scale * offset
        return new_ldmk

    def get_center(self, ldmk):
        xmax = np.max(ldmk[:, 0])
        xmin = np.min(ldmk[:, 0])

        ymax = np.max(ldmk[:, 1])
        ymin = np.min(ldmk[:, 1])

        xc = (xmax + xmin) / 2.0
        yc = (ymax + ymin) / 2.0
        return np.asarray([xc, yc])

    def draw_heatmap_from_68_landmark(self, lmk, width=512, height=512, draw_mouth=True):
                                                                                                   
        heatmap = np.zeros((width, height), dtype=np.uint8)

        def convert(tuple):
            return (int(tuple[0]), int(tuple[1])) 

        def draw_line(list):                                                                                                                              
            for i in range(len(list)-1):                                                                                                                   
                #print(lmk[list[i]])                                                                                                                        
                cv2.line(heatmap, convert(lmk[list[i]]), convert(lmk[list[i+1]]), thickness=2, color=255)                                                                                                                                                       
        def draw_circle(list):                                                                                                                                                                                                                                    
            for i in range(len(list)):
                if i != len(list)-1:
                    cv2.line(heatmap, convert(lmk[list[i]]), convert(lmk[list[i+1]]), thickness=2, color=255)
                else:
                    cv2.line(heatmap, convert(lmk[list[i]]), convert(lmk[list[0]]), thickness=2, color=255)
        if draw_mouth:
            '''draw mouth outter'''                                                                                                                      
            mo_list = list(range(48, 60))                                                                             
            draw_circle(mo_list)

            '''draw mouth inner'''                                                                                                                       
            mi_list = list(range(60, 68))                                                                          
            draw_circle(mi_list)                                                                                                                                 

        '''draw left eye'''                                                                                                                      
        le_list = list(range(36, 42))                                                                             
        draw_circle(le_list)

        '''draw right eye'''                                                                                                                      
        re_list = list(range(42, 48))                                                                             
        draw_circle(re_list)

        '''draw left eye brow'''                                                                                                                      
        leb_list = list(range(17,22))                                                                             
        draw_line(leb_list)

        '''draw right eye brow'''                                                                                                                      
        reb_list = list(range(22,27))                                                                             
        draw_line(reb_list)

        '''draw nose'''                                                                                                                      
        ns_list1 = list(range(27, 31))                                                                            
        draw_line(ns_list1)
        ns_list2 = list(range(31, 36))                                                                            
        draw_line(ns_list2)

        '''draw jaw'''                                                                                                                      
        jaw_list = list(range(0,17))                                                                             
        draw_line(jaw_list)
                                                                                                                                        
        #heatmap = cv2.GaussianBlur(heatmap, ksize=(5, 5), sigmaX=1, sigmaY=1)
        #print('heatmap size:', heatmap.shape)                                                                              
        return heatmap 