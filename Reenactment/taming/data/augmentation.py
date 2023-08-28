import numpy as np
import random
import math
import cv2
import torch


class RandomMaskGenerator:
    ## We only mask half face
    def __init__(self, size=256, patch_size=16, isTrain=False, num_masking_patches=88, 
                    min_num_patches=4, max_num_patches=None,
                    min_aspect=0.3, max_aspect=None, patch=None):
        self.size = size
        self.patch_size = patch_size
        input_size = size // patch_size

        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size
        self.height = self.height // 2

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        self.isTrain = isTrain

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def get_shape(self):
        return self.height, self.width

    def reset_seed(self,):
        fix_seed = np.random.randint(2147483647)
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)

    def random_retangle_mask(self, mask, mask_count):
        # mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        # mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask

    def __call__(self, is_ref=False, is_half=False):
        # self.reset_seed()
        if np.random.uniform() < 0.25 or self.isTrain:
            mask = np.ones(shape=(256,256))
            #patch = 16
            patch= 64
            #mask[128:, patch:-patch] = 0
            if not is_half:
                mask[128:224, patch:-patch] = 0
            else:
                mask[128:, :] = 0

            mask = 1 - mask
        else:
            upper_mask = np.zeros(shape=(self.height, self.width), dtype=np.int)
            lower_mask = np.zeros(shape=(self.height, self.width), dtype=np.int)

            mouth_radius = 6
            lower_mask[:mouth_radius, self.width//2-mouth_radius//2:self.width//2+mouth_radius//2] = 1
            lower_mask_count = np.sum(lower_mask)
            mask = self.random_retangle_mask(lower_mask, lower_mask_count)
            mask = np.vstack([upper_mask, lower_mask])

            #mask = 1 - cv2.resize(mask, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if is_ref:
            mask = np.flipud(mask)
        return mask


if __name__ == '__main__':
    random_mask_generator = RandomMaskGenerator(isTrain=True)
    for i in range(10):
        print(np.unique(random_mask_generator()))
        cv2.imwrite('{}.png'.format(i), (random_mask_generator()*255.).astype(np.uint8))