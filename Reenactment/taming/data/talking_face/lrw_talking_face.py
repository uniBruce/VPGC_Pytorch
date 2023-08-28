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


class LRWTalkingFaceDataset(Dataset):
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

        self.audio_lmdb_paths = args.audio_lmdb_paths
        self.img_lmdb_paths = args.img_lmdb_paths

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

    def transform_seq(self, imgs, seed, is_ref=False, is_aug=False):
        # import pdb; pdb.set_trace();
        # fix_seed = np.random.randint(2147483647)
        res_images = []
        res_masks = []
        for img in imgs:
            img = self.preprocessor(image=img)['image']
            random.seed(seed)
            torch.manual_seed(seed)
            if is_aug:
                img = self.vqaugmentor(image=img)['image']
            else:
                img = self.augmentor(image=img)['image']
            mask = self.random_mask_generator(is_ref)
            res_images.append(torch.from_numpy(np.array(img)).unsqueeze(0))
            res_masks.append(torch.from_numpy(np.array(mask)).unsqueeze(0))

        res_image = (torch.cat(res_images, 0) / 127.5 - 1.0).float()
        res_mask = torch.cat(res_masks, 0).float()

        return res_image, res_mask


    #     res = torch.utils.data._utils.collate.default_collate(res) # [bs,clip_len,c,h,w]
    #     res_image = (res['image']/127.5 - 1.0).float()
    #     res_mask = (res['mask']).float()
    #     # self.reset_seed()

    #     return res_image, res_mask

    def get_align_M(self, opt):
        assert self.opt.datatype == 'vox2'
        src = np.array([[86.2998,  58.486058],
                        [135.61,  57.8444],
                        [111.4154, 116.2026571]])
    
        dst = np.array([[91.,124.],
                        [165.,124.],
                        [128.,200.]])*224./256.
        M = get_affine(src, dst)
        return M

    def get_train_clips(self, num_frames, audio=False):
        """
        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        num_frames = int(num_frames)
        ori_clip_len = self.opt.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 14 + self.num_inputs:
            base_offsets = np.arange(self.num_clips) * avg_interval + 8 + self.num_inputs
            clip_offsets = base_offsets + np.random.randint(
                avg_interval - 8 - self.num_inputs - 6, size=self.num_clips)
        else:
            clip_offsets = np.ones((self.num_clips, )).astype(np.int) * 2

        frame_inds = clip_offsets[:, None] + np.arange(
            self.opt.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        frame_inds = np.mod(frame_inds, num_frames)

        if audio:
            start_lower = max(0, frame_inds[0] - 20)
            end_higher = min(frame_inds[-1] + 20, num_frames)
        else:
            start_lower = 0
            end_higher = num_frames

        lower_bound = max(1, frame_inds[0] - 1)
        upper_bound = min(num_frames - 1, frame_inds[-1] + 1)
        selection_list = np.concatenate([np.arange(start_lower, lower_bound), np.arange(upper_bound, end_higher)], 0)
        input_frame_inds = np.random.choice(selection_list, self.num_inputs)
        # print(selection_list, input_frame_inds)

        return frame_inds, input_frame_inds

    def frame2audio_indexs(self, frame_inds):
        start_frame_ind = frame_inds - self.opt.num_frames_per_clip // 2

        start_audio_inds = start_frame_ind * self.opt.num_bins_per_frame
        return start_audio_inds

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

    def load_spectrograms(self, frame_inds, mel):
        spectrograms = []
        num_clips = self.num_clips

        mel_shape = mel.shape

        frame_inds = frame_inds.reshape(num_clips, -1)

        for i in range(num_clips):
            frame_indices = frame_inds[i]
            start_audio_inds = self.frame2audio_indexs(frame_indices)

            for k, audio_ind in enumerate(start_audio_inds):

                if (audio_ind + self.num_audio_bins) <= mel_shape[0] and audio_ind >= 0:
                    spectrogram = np.array(mel[audio_ind:audio_ind + self.num_audio_bins, :]).astype('float32')
                else:
                    print('(audio_ind {} + self.num_audio_bins {}) > mel_shape[0] {} num_frame: {}'.format(
                        audio_ind, self.num_audio_bins, mel_shape[0], self.num_frames))

                    if k > 0:
                        spectrogram = spectrograms[k-1]
                    else:
                        spectrogram = np.zeros((self.num_audio_bins, mel_shape[1])).astype(np.float32)

                spectrograms.append(spectrogram)

        spectrograms = np.stack(spectrograms, 0)

        spectrograms = torch.from_numpy(spectrograms)
        spectrograms = spectrograms.unsqueeze(1)

        spectrograms = spectrograms.transpose(-2, -1)
        return spectrograms

    def load_audio_wav(self, frame_inds, data):
        res = []
        num_clips = self.num_clips

        data_shape = data.shape

        frame_inds = frame_inds.reshape(num_clips, -1)

        for i in range(num_clips):
            frame_indices = frame_inds[i]
            start_audio_inds = self.frame2audio_indexs(frame_indices)

            for k, audio_ind in enumerate(start_audio_inds):

                if (audio_ind + self.num_audio_bins) <= data_shape[0] and audio_ind >= 0:
                    spectrogram = np.array(data[audio_ind:audio_ind + self.num_audio_bins]).astype('float32')
                else:
                    print('(audio_ind {} + self.num_audio_bins {}) > data_shape[0] {} num_frame: {}'.format(
                        audio_ind, self.num_audio_bins, data_shape[0], self.num_frames))

                    if k > 0:
                        spectrogram = res[k-1]
                    else:
                        spectrogram = np.zeros((self.num_audio_bins)).astype(np.float32)

                res.append(spectrogram)

        res = np.stack(res, 0)

        return res       

    def load_frames_pil(self, image_reader, frame_inds, video_key, is_ref=False):
        imgs = []
        if frame_inds.ndim != 1:
            frame_inds = np.squeeze(frame_inds)

        for frame_idx in frame_inds:
            if not is_ref:
                if frame_idx < 2:
                    frame_idx += 2
            frame_key = video_key + '_' + '{:06d}'.format(frame_idx)
            img = self.load_img_pil(image_reader, frame_key)
            imgs.append(img)

        return imgs

    def load_frames(self, frame_inds, video_path, use_augmentation=False, crop=True):
        database, video_key = video_path
        imgs = []
        if frame_inds.ndim != 1:
            frame_inds = np.squeeze(frame_inds)

        for frame_idx in frame_inds:
            # temporary solution for frame index offset.
            # TODO: add offset attributes in datasets.
            if frame_idx < 2:
                frame_idx += 2
            frame_key = video_key + '_' + '{:06d}'.format(frame_idx)

            img = self.load_img(database, frame_key, crop=crop)
            imgs.append(img)

        imgs = np.stack(imgs)

        target_imgs = self.to_Tensor(imgs)

        return target_imgs


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
        frame_meta_info = img_reader.get_item(local_index)
        
        video_key, word_label = frame_meta_info
        audio_key = '_'.join(video_key.split('_')[1:])

        num_frames = 29
        self.num_frames = num_frames

        target_frame_inds, ref_frame_ind = self.get_train_clips(num_frames, audio=True)
        # target_frame_inds, input_frame_ind = self.get_train_clips(num_frames, audio=True)
        # print(index, target_frame_inds, ref_frame_ind)

        target_frames = self.load_frames_pil(img_reader, target_frame_inds, video_key)
        
        # tgt_imgs = self.transform_seq(target_frames)
        # data_dict = {'image': tgt_imgs[0]}

        seed = random.randint(0, 2 ** 32)

        tgt_imgs, tgt_masks = self.transform_seq(target_frames, seed, is_aug=True)
        
        
        # tgt_masks[0].save('tgt_mask.jpg')
        data_dict = {
                    # 'image': tgt_imgs[0],
                    # 'mask': tgt_masks[0],
                    'image': tgt_imgs,
                    'mask': tgt_masks,
                    'clip_len': self.clip_len
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
        

        if self.opt.load_aug:
            seed = random.randint(0, 2 ** 32)
            aug_imgs, _ = self.transform_seq(target_frames, seed, is_aug=True)
            data_dict['aug_image'] = aug_imgs

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