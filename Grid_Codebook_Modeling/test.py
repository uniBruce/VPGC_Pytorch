import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import argparse
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from taming.data.talking_face.obama_dataset import ObamaDataset
from taming.models.vqgan import VQModel
from taming.models.vqgan_slidinggrid import VQGridModel as Model

from taming.util import dict_unite, prepare_sub_folder, write_image, save_torch_img
from skimage.io import imsave
import sys
import tqdm
import yaml
from torchvision import transforms

parser = argparse.ArgumentParser()
# for base options
parser.add_argument('--load_path', type=str, default='../Pretrained/taming/Obama/N-Step-Checkpoint_21_270000.ckpt', help='load path')
parser.add_argument('--modelbase', type=str, default='configs/obama_sliding_4.yaml', help='configure files')
parser.add_argument('--database', type=str, default='configs/talking_config/lrw_test.yaml', help='configure files')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--batch_internal', type=int, default=4, help='internal for consecutive frames')
parser.add_argument('--output_path', type=str, default='Results', help='output directory for generated results')
parser.add_argument('--model_name', type=str, default='Obama', help='output directory for generated results')
parser.add_argument('--gpu', type=str, default='0', help='gpu to use')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


test_dataset = ObamaDataset(opt.database)
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=2)

#trainer = Seq2Face_Model(opt)
f = open(opt.modelbase, 'r')
config = yaml.load(f, Loader=yaml.FullLoader)
#print(config)
config = edict(config)
ddconfig = config.model.params.ddconfig
lossconfig = config.model.params.lossconfig
trainer = Model(ddconfig, lossconfig, config.model.params.n_embed, config.model.params.embed_dim, ckpt_path=opt.load_path)
trainer.cuda()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
else:
    print("Let's use 1 GPU")

image_directory = prepare_sub_folder(os.path.join(opt.output_path, opt.model_name))

### Load Network Weights

print('----Begin Testing----')


for i, data in enumerate(test_loader):
    ### testing code
    fake_img = trainer.inference(data)
    fake_seq = trainer.batch2seq(fake_img, opt.batch_internal, infer=True)
    #toPIL = transforms.ToPILImage()

    ### sequential generate each frame in a batch
    for j in range(len(fake_seq)):
        save_path = os.path.join(image_directory, "%04d.jpg"%(i*opt.batch_internal + j))
        save_torch_img(fake_seq[j][0], save_path)

command = 'ffmpeg -r 60 -f image2 -i {folder}/%04d.jpg  -vcodec libx264  {folder}/{name}.mp4'.format(folder=image_directory, name=opt.model_name)
os.system(command)
print('The Generated Video is saved in %s !'%image_directory)
        



    

