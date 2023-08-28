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


class LRWMouthDataset(Dataset):
    def __init__(self, data_config_path, transform=None, data_root=None, phase='train', 
                       im_preprocessor_config=None, sample_mode='list', main_args=None, 
                       with_mask=False, random_mask_config=None):
        with open(data_config_path, 'r') as f:
            args = yaml.load(f,Loader=yaml.FullLoader)
            args = edict(args)
            args = args.dataset
        args = edict(args)
        if main_args is not None:
            args.update(edict(vars(main_args)))
            
        # import pdb;pdb.set_trace();
        self.opt = args
        self.with_mask = with_mask

        if transform is None:
            self.init_transform()
        else:
            raise ValueError
        # import pdb;pdb.set_trace();
        self.clip_len = self.opt.clip_len
        self.frame_interval = self.opt.frame_interval
        self.num_clips = self.opt.num_clips
        self.num_inputs = 1

        self.img_lmdb_paths = args.img_lmdb_paths
        self.audio_lmdb_paths = args.audio_lmdb_paths if self.opt.load_audio else None
        self.spec_lmdb_paths = args.spec_lmdb_paths if self.opt.load_spec else None

        if self.opt.load_audio:
            print('************ load audio wav **************')
            self.audio_lmdb_readers = [LMDBReader(p, debug=False, mode='audio') for p in self.audio_lmdb_paths] 
        elif self.opt.load_spec:

            print('************ load spectrograms **************')
            self.spec_lmdb_readers = [LMDBReader(p, debug=False, mode='audio') for p in self.spec_lmdb_paths]
        self.img_lmdb_readers = [LMDBReader(p, debug=False, mode='visual') for p in self.img_lmdb_paths]
        # we need to mannually make sure the correspondance
        
        self.dataset_size = sum([len(reader_i) for reader_i in self.img_lmdb_readers])
        idx2img_reader = [[i,]*len(reader_i) for i, reader_i in enumerate(self.img_lmdb_readers)]
        self.idx2img_reader = []
        self.item_cnt = []
        for i, idx in enumerate(idx2img_reader):
            self.idx2img_reader += idx
            self.item_cnt += [len(self.img_lmdb_readers[i]),]
        self.item_cnt_cumsum = np.cumsum(np.array(self.item_cnt))


        if self.opt.load_audio:
            self.opt.num_bins_per_frame = 16000 // 25
            from transformers import Wav2Vec2FeatureExtractor
            # self.audio_feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base")
            if 'wav2vec_path' in self.opt and self.opt.wav2vec_path:
                self.audio_feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path=self.opt.wav2vec_path)
            else:
                self.audio_feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path='wav2vec2-base')

        elif self.opt.load_spec:
            self.opt.num_bins_per_frame = 4

        self.num_audio_bins = self.opt.num_frames_per_clip * self.opt.num_bins_per_frame
        
        # use the vq-talking mask
        self.random_mask_generator = None
        if random_mask_config is not None and with_mask is True:
            self.random_mask_generator = instantiate_from_config(random_mask_config)
        self.get_mask()


    def get_mask(self):
        self.mask = self.random_mask_generator()
        self.mask = Image.fromarray(self.mask)

    def init_transform(self):
        # self.rescaler = albumentations.SmallestMaxSize(max_size = self.opt.crop_size)
        # if not self.random_crop:
        #     self.cropper = albumentations.CenterCrop(height=self.opt.crop_size,width=self.opt.crop_size)
        # else:
        # self.cropper = albumentations.RandomCrop(height=self.opt.crop_size,width=self.opt.crop_size)
        self.cropper = albumentations.Crop(x_min=16, y_min=0, x_max=256-16, y_max=256-32, always_apply=True)
        self.rescalar = albumentations.Resize(height=256, width=256)

        self.preprocessor = albumentations.Compose([self.cropper, self.rescalar])

        self.augmentor = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Rotate(limit=20),
            # albumentations.pytorch.transforms.ToTensorV2()
            # albumentations.IAAPerspective()
        ])

        self.vqaugmentor = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.IAAPerspective(scale=(0.02, 0.1)),
            albumentations.Rotate(limit=20),
            #albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            
        ])
        # return self.preprocessor, self.augmentor

    def reset_seed(self):
        fix_seed = np.random.randint(2147483647)
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)

    def transform_single(self, img, seed, is_ref=False, is_half=False, is_aug=False, load_mask=False):
        # import pdb; pdb.set_trace();
        # fix_seed = np.random.randint(2147483647)
        mask = self.random_mask_generator(is_half = is_half)
        img = self.preprocessor(image=img)['image']
        if load_mask:
            img = np.expand_dims(mask,2) * img
        random.seed(seed)
        torch.manual_seed(seed)

        if is_aug:
            img = self.vqaugmentor(image=img)['image']
        else:
            img = self.augmentor(image=img)['image']
        
        res_image = (torch.from_numpy(np.array(img)) / 127.5 - 1.0).float()
        res_mask = torch.from_numpy(np.array(mask)).float()

        return res_image, res_mask


    #     res = torch.utils.data._utils.collate.default_collate(res) # [bs,clip_len,c,h,w]
    #     res_image = (res['image']/127.5 - 1.0).float()
    #     res_mask = (res['mask']).float()
    #     # self.reset_seed()

    #     return res_image, res_mask

    def load_img_pil(self, img_reader, frame_key):
        img_pil = img_reader.read_key(frame_key)
        image_np = np.array(img_pil).astype(np.uint8)

        return image_np

    def to_Tensor(self, img):
        if img.ndim == 3:
            wrapped_img = img.transpose(2, 0, 1)
        elif img.ndim == 4:
            wrapped_img = img.transpose(0, 3, 1, 2)
        else:
            wrapped_img = img
        wrapped_img = torch.from_numpy(wrapped_img).float()

        return wrapped_img 

    def sample_new_index(self, index):
        index_start = int(index/self.opt.data_ratio_per_epoch)
        index_end = int((index+1)/self.opt.data_ratio_per_epoch)

        index = np.random.randint(index_start, index_end)
        index = min(self.dataset_size, index)
        return index

    def __getitem__(self, index):
        index = self.sample_new_index(index)
        index = index % self.dataset_size
        return self.get_item(index)

    def get_item(self, index):
        img_reader_idx = self.idx2img_reader[index]
        img_reader = self.img_lmdb_readers[img_reader_idx]

        local_index = index if img_reader_idx <= 0 else index - self.item_cnt_cumsum[img_reader_idx-1]
        frame_meta_info = img_reader.get_key_name(local_index)
        #print('frame info:', frame_meta_info)
        #video_key, frame_inx = frame_meta_info

        target_frame = self.load_img_pil(img_reader, frame_meta_info)
    

        seed = random.randint(0, 2 ** 32)

        #tgt_imgs, tgt_masks = self.transform_seq(target_frames, seed)
        tgt_img, tgt_mask = self.transform_single(target_frame, seed, is_aug=self.opt.load_aug, is_half=self.opt.load_half, load_mask=True)
        
        # tgt_masks[0].save('tgt_mask.jpg')
        data_dict = {
                    # 'image': tgt_imgs[0],
                    # 'mask': tgt_masks[0],
                    'image': tgt_img,
                    'mask': tgt_mask
                    #'clip_len': self.clip_len
                     }


        if self.opt.load_word_label:
            data_dict['word_label'] = int(word_label)

        if self.opt.load_ref:
            ref_frames = self.load_frames_pil(img_reader, ref_frame_ind, video_key, is_ref=True)
            ref_imgs, ref_masks = self.transform_seq(ref_frames, seed, is_ref=True)
            data_dict['ref_image'] = ref_imgs[0]
            data_dict['ref_mask'] = ref_masks[0]
            # data_dict['ref_image'] = ref_imgs
            # data_dict['ref_mask'] = ref_masks
        

        # if self.opt.load_aug:
        #     seed = random.randint(0, 2 ** 32)
        #     aug_imgs, _ = self.transform_seq(target_frames, seed, is_aug=True)
        #     data_dict['aug_image'] = aug_imgs

        if self.opt.load_audio:
            audio = self.audio_lmdb_readers[0].read_key(audio_key)
            audio_std = self.audio_feat_extractor(audio, return_tensors="pt", sampling_rate=16000).input_values[0]
            ori_len, cur_len = self.opt.num_bins_per_frame*29, len(audio_std)
            cut_len = cur_len - ori_len
            # cut_len = int((cur_len-ori_len)/cur_len * len(audio_std))
            audio_std = audio_std[int(cut_len*0.4):int(cut_len*0.4)+ori_len]

            audio_stds = self.load_audio_wav(target_frame_inds, audio_std)
            data_dict['audio'] = audio_stds

        if 'load_spec' in self.opt and self.opt.load_spec:
            mel = self.spec_lmdb_readers[0].read_key(audio_key)
            mel = mel[6:-7] # mel 129 -> 116
            spectrograms = self.load_spectrograms(target_frame_inds, mel)
            data_dict['spectrograms'] = spectrograms

        return data_dict

    def __len__(self):
        return int(self.dataset_size*self.opt.data_ratio_per_epoch)