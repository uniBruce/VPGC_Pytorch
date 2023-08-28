import os
import numpy as np
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from taming.data.talking_face.lrw_talking_face import LRWTalkingFaceDataset
from taming.data.talking_face.lrw_mouth import LRWMouthDataset
from taming.data.talking_face.obama_dataset import ObamaDataset

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class LRWTrain(CustomBase):
    def __init__(self, size, training_images_list_file, with_mask=False, random_mask_config=None):
        super().__init__()
        #self.data = LRWTalkingFaceDataset(training_images_list_file, with_mask=with_mask, random_mask_config=random_mask_config)
        # self.data = LRWMouthDataset(training_images_list_file, with_mask=with_mask, random_mask_config=random_mask_config)
        self.data = ObamaDataset(training_images_list_file)


class LRWTest(CustomBase):
    def __init__(self, size, test_images_list_file, with_mask=False, random_mask_config=None):
        super().__init__()
        #self.data = LRWTalkingFaceDataset(test_images_list_file, with_mask=with_mask, random_mask_config=random_mask_config)
        # self.data = LRWMouthDataset(test_images_list_file, with_mask=with_mask, random_mask_config=random_mask_config)
        self.data = ObamaDataset(test_images_list_file)
