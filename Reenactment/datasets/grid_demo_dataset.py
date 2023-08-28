import os
from datasets.base_dataset import BaseDataset
import os.path
import glob
from pathlib import Path
import torch
from skimage.io import imread, imsave
from util import util
from PIL import Image
import bisect
import numpy as np
import io
import cv2
import random
import h5py
import scipy.io as scio
import albumentations.pytorch


class GridDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)

        if self.opt.isTrain:
            self.images_root = os.path.join(self.opt.data_root, 'images')
            self.images_dir = os.path.join(self.images_root, 'train')
            self.ldmk_root = os.path.join(self.opt.data_root, 'heatmap')
            self.ldmks_dir = os.path.join(self.ldmk_root, 'train')
            self.images = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))[:] # 49800 for train
        else:
            self.ldmks_dir = os.path.join(self.opt.data_root, 'heatmap/test')

        if self.opt.isTrain:
            self.ldmks = sorted(glob.glob(os.path.join(self.ldmks_dir, '*.txt')))
        else:
            self.ldmks = sorted(glob.glob(os.path.join(self.ldmks_dir, '*.jpg')))[:1000]

        self.len_file = len(self.ldmks) // self.opt.batch_internal
    

        print('Loading data from %s, total number of images is %d'%(self.ldmks_dir, len(self.ldmks))) 

    def __getitem__(self, index):
        #self.heat_gt_path = self.heat_gts[index]
        # if self.opt.use_ref:
        #     tmp =[]
        #     random_idx = random.sample(range(0, len(self.images)), 4)
        #     for j in range(4):
        #         output = imread(self.images[random_idx[j]])
        #         output = albumentations.pytorch.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})(image=output)['image']
        #         tmp.append(output)
        #     self.full_cand = torch.from_numpy(np.concatenate(tmp))
        
        batch_img = []
        batch_heat = []
        if self.opt.isTrain:
            scale = random.uniform(0.95, 1.05)
            #scale = 1.0
        else:
            scale = 1.0
        for i in range(self.opt.batch_internal):
            idx = index*self.opt.batch_internal+i
            if self.opt.isTrain:
                self.image = self.get_image(self.images[idx])
                batch_img.append(self.image)
            ### Load Adajacent Heatmap (3 in total) for Smoothing
            #self.heatmap = self.get_sevenset(self.heatmaps, idx, repeat=self.opt.repeat_heat)
            self.ldmk = self.get_sevenset(self.ldmks, idx, scale=scale, repeat=self.opt.repeat_heat)
            batch_heat.append(self.ldmk)
            
        if self.opt.isTrain:
            self.input_image = torch.cat(batch_img, 0)
        self.feature_map = torch.cat(batch_heat, 0)
        
        if self.opt.isTrain:
            return_list = {'feature_map': self.feature_map, 'tgt_image': self.input_image} #'heat_gt':self.heat_gt
        else:
            return_list = {'feature_map': self.feature_map}          
        return return_list

    def __len__(self):
        self.len_file = len(self.ldmks) // self.opt.batch_internal
        return self.len_file

    def get_image(self, path, scale=1.0, is_heat=False):
        if is_heat:
            if 'jpg' in path:
                image = imread(path)
            elif 'txt' in path:
                ldmk = np.loadtxt(path).astype(np.float)
                image = util.draw_heatmap_from_68_landmark(ldmk)
            elif 'mat' in path:
                mat = scio.loadmat(path)
                raw_ldmk = mat['lm68'][0]
                align = mat['align_m']
                ldmk = self.recover_lmk(align, raw_ldmk)
                ldmk = self.get_scale(ldmk, scale)
                image = util.draw_heatmap_from_86_landmark(ldmk)
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image.astype(np.float32)/255)
        else:
            image = imread(path)
            image = albumentations.pytorch.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})(image=image)['image']
        return image

    def get_sevenset(self, images, idx, scale=1.0, repeat=False):
        curr_path = images[idx]
        if repeat:
            prev2_path = images[idx]
            prev1_path = images[idx]
            prev_path = images[idx]
            next_path = images[idx]
            next1_path = images[idx]
            next2_path = images[idx]
        else:
            if idx > 2 and idx < (self.len_file * self.opt.batch_internal-3):
                prev2_path = images[idx-3]
                prev1_path = images[idx-2]
                prev_path = images[idx-1]
                next_path = images[idx+1]
                next1_path = images[idx+2]
                next2_path = images[idx+3]
            elif idx == 0:
                prev2_path = images[idx]
                prev1_path = images[idx]
                prev_path = images[idx]
                next_path = images[idx+1]
                next1_path = images[idx+2]
                next2_path = images[idx+3]
            elif idx == 1:
                prev2_path = images[idx-1]
                prev1_path = images[idx-1]
                prev_path = images[idx-1]
                next_path = images[idx+1]
                next1_path = images[idx+2]
                next2_path = images[idx+3]
            elif idx == 2:
                prev2_path = images[idx-2]
                prev1_path = images[idx-2]
                prev_path = images[idx-1]
                next_path = images[idx+1]
                next1_path = images[idx+2]
                next2_path = images[idx+3]
            elif idx == (self.len_file * self.opt.batch_internal-3):
                prev2_path = images[idx-3]
                prev1_path = images[idx-2]
                prev_path = images[idx-1]
                next_path = images[idx+1]
                next1_path = images[idx+2]
                next2_path = images[idx+2]
            elif idx == (self.len_file * self.opt.batch_internal-2):
                prev2_path = images[idx-3]
                prev1_path = images[idx-2]
                prev_path = images[idx-1]
                next_path = images[idx+1]
                next1_path = images[idx+1]
                next2_path = images[idx+1]
            elif idx == (self.len_file * self.opt.batch_internal-1):
                prev2_path = images[idx-3]
                prev1_path = images[idx-2]
                prev_path = images[idx-1]
                next_path = images[idx]
                next1_path = images[idx]
                next2_path = images[idx]
    
        curr_img = self.get_image(curr_path, scale, is_heat=True)
        prev2_img = self.get_image(prev2_path, scale, is_heat=True)
        prev1_img = self.get_image(prev1_path, scale, is_heat=True)
        prev_img = self.get_image(prev_path, scale, is_heat=True)
        next_img = self.get_image(next_path, scale, is_heat=True)
        next1_img = self.get_image(next1_path, scale, is_heat=True)
        next2_img = self.get_image(next2_path, scale, is_heat=True)
        comb = torch.cat([prev2_img, prev1_img, prev_img, curr_img, next_img, next1_img, next2_img], 0)
        return comb


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

    def recover_lmk(self, M, lmk):
        #### np.matmul(align_m[:2, :2], lmk) + align_m[:2, 2:3]
        lmk = lmk.transpose(1, 0) # [2, 86]
        lmk = lmk - M[:2, 2:3]
        M_i = np.matrix(M).I
        lmk = np.matmul(M_i[:2, :2], lmk)
        lmk = lmk.transpose(1, 0)
        return np.asarray(lmk)






