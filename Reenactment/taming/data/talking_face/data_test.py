from lmdb_utils import LMDBReader
import numpy as np

img_lmdb_paths = ['/root/lrw/imgs/0_100/',
                '/root/lrw/imgs/100_200/'
                #'/root/lrw/imgs/200_300/',
                #'/root/lrw/imgs/300_400/',
                #'/root/lrw/imgs/400_500/'
                ]

img_lmdb_readers = [LMDBReader(p, debug=False, mode='visual') for p in img_lmdb_paths]
        # we need to mannually make sure the correspondance
        
dataset_size = sum([len(reader_i) for reader_i in img_lmdb_readers])
idx2img_reader = [[i,]*len(reader_i) for i, reader_i in enumerate(img_lmdb_readers)]
idx2img_list = []
item_cnt = []
for i, idx in enumerate(idx2img_reader):
    idx2img_list += idx
    item_cnt += [len(img_lmdb_readers[i]),]
item_cnt_cumsum = np.cumsum(np.array(item_cnt))

#print(idx2img_list)
print(dataset_size, item_cnt)

index = 120000
img_reader_idx = idx2img_list[index]
print(img_reader_idx)
img_reader = img_lmdb_readers[img_reader_idx]
local_index = index if img_reader_idx <= 0 else index - item_cnt_cumsum[img_reader_idx-1]
print(local_index)
frame_key = img_reader.get_key_name(local_index)
img_pil = img_reader.read_key(frame_key)
image_np = np.array(img_pil).astype(np.uint8)
print(image_np.shape)