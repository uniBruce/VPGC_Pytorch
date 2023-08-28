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
            self.ldmk_root = os.path.join(self.opt.data_root, 'comb')
            self.ldmks_dir = os.path.join(self.ldmk_root, 'train')
            self.images = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))[:] # 49800 for train
        else:
            self.ldmks_dir = os.path.join(self.opt.data_root, 'comb/test')

        if self.opt.isTrain:
            self.ldmks = sorted(glob.glob(os.path.join(self.ldmks_dir, '*.txt')))
        else:
            self.ldmks = sorted(glob.glob(os.path.join(self.ldmks_dir, '*.txt')))[:1000]

        self.len_file = len(self.ldmks) // self.opt.batch_internal
    

        print('Loading data from %s, total number of images is %d'%(self.ldmks_dir, len(self.ldmks))) 

    def __getitem__(self, index):
        
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
                data = np.loadtxt(path).astype(np.float)
                data = self.get_scale(data, scale)
                ldmk = data[:68, :]
                mat = data[68:, :]
                image = draw_heatmap_from_68_landmark(ldmk)
                image = draw_heatmap_from_86_landmark(mat, heatmap=image)
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




def convert(tuple):
    return (int(tuple[0]), int(tuple[1])) 

def draw_heatmap_from_68_landmark(lmk, heatmap=None, width=512, height=512, draw_mouth=False):
    if heatmap is None:                                                                                         
        heatmap = np.zeros((width, height), dtype=np.uint8)

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

    # '''draw nose'''                                                                                                                      
    # ns_list1 = list(range(27, 31))                                                                            
    # draw_line(ns_list1)
    # ns_list2 = list(range(31, 36))                                                                            
    # draw_line(ns_list2)

    # '''draw jaw'''                                                                                                                      
    # jaw_list = list(range(0,17))                                                                             
    # draw_line(jaw_list)
                                                                                                                                      
    #heatmap = cv2.GaussianBlur(heatmap, ksize=(5, 5), sigmaX=1, sigmaY=1)
    #print('heatmap size:', heatmap.shape)                                                                              
    return heatmap 

def draw_heatmap_from_86_landmark(lmk, heatmap=None, width=512, height=512, draw_mouth=True):
    if heatmap is None:                                                                                              
        heatmap = np.zeros((width, height), dtype=np.uint8)

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
        mo_list = [66, 72, 67, 68, 69, 73, 70, 74, 85, 71, 84, 75]                                                                             
        draw_circle(mo_list)

        '''draw mouth inner'''                                                                                                                       
        mi_list = [76, 78, 79, 80, 77, 83, 82, 81]                                                                          
        draw_circle(mi_list)                                                                                                                                 

    # '''draw left eye'''                                                                                                                      
    # le_list = [35, 36, 41, 37, 38, 39, 42, 40]                                                                             
    # draw_circle(le_list)

    # '''draw right eye'''                                                                                                                      
    # re_list = [43, 44, 49, 45, 46, 47, 50, 48]                                                                             
    # draw_circle(re_list)

    # '''draw left eye brow'''                                                                                                                      
    # leb_list = [17, 18, 19, 20, 21, 25, 24, 23, 22]                                                                             
    # draw_circle(leb_list)

    # '''draw right eye brow'''                                                                                                                      
    # reb_list = [26, 27, 28, 29, 30, 34, 33, 32, 31]                                                                             
    # draw_circle(reb_list)

    '''draw nose'''                                                                                                                      
    ns_list1 = [51, 52, 53, 54]                                                                           
    draw_line(ns_list1)
    ns_list2 = [57, 59, 61, 62, 63, 64, 65, 60, 58]                                                                            
    draw_line(ns_list2)

    '''draw jaw'''                                                                                                                      
    jaw_list = list(range(0,17))                                                                             
    draw_line(jaw_list)
                                                                                                                                      
    #heatmap = cv2.GaussianBlur(heatmap, ksize=(5, 5), sigmaX=1, sigmaY=1)
    #print('heatmap size:', heatmap.shape)                                                                              
    return heatmap