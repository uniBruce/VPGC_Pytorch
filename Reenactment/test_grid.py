import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import argparse
from torch.utils.data import DataLoader

from datasets.grid_dataset import GridDataset
from models.grid_seq_final import GridFormer7_Model

from util import util, distributed
from torch.utils.data.distributed import DistributedSampler
import sys
import tqdm
import tensorboardX


parser = argparse.ArgumentParser()
parser.add_argument('--size', default = 'normal', type = str)
parser.add_argument('--Dataset_type', default='F5Dataset', type=str, help='type of dataset')
parser.add_argument('--balance', default = False, type = bool, help='whether use balanced strategy')
parser.add_argument('--isTrain', default = False, type = bool, help='whether training or not')
parser.add_argument('--fp16', default = False, type = bool, help='whether training or not')
parser.add_argument('-m', '--model_name', default='Rodgers_seq_prior_1.0', type=str, help='name of training model')
parser.add_argument('-r', '--resume', default=False, type=bool, help='whether to resume training')
parser.add_argument('--data_root', default='Dataset/Obama', type=str, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=1, help='number of sata samples for each iteration')
parser.add_argument('--batch_internal', type=int, default=4, help='number of sata samples for each iteration')
parser.add_argument('--vis_freq', type=int, default=100, help='number of sata samples for each iteration')
parser.add_argument('--print_freq', type=int, default=10, help='number of sata samples for each iteration')
parser.add_argument('-o', '--output_path', default='./Results', type=str, help='path to the checkpoint files and predicted images')
parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of to save the latest results(iterations)')
parser.add_argument('--save_epoch_freq', type=int, default=3, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--continue_train', default=False, action='store_true', help='continue training: load the latest model')        
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--load_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--n_epochs_warm_up', type=int, default=5, help='number of epochs warm up')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--max_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--n_epochs_decay', type=int, default=10, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=5000, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--lr_decay_gamma', type=float, default=0.8, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--TTUR', action='store_true', help='Use TTUR training scheme')        
parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|original|hinge)')
parser.add_argument('--pool_size', type=int, default=1, help='the size of image buffer that stores previously generated images')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
parser.add_argument('--frame_jump', type=int, default=1, help='jump frame for training, 1 for not jump')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--seq_max_len', type=int, default=120, help='maximum sequence clip frames sent to network per iteration')
parser.add_argument('--gpu_id', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')  
parser.add_argument('--load_path', type=str, default='Pretrained/Grid_Codebook/Obama/N-Step-Checkpoint_19_120000.ckpt', help='Obama model')
parser.add_argument('--model_base', type=str, default='configs/obama_sliding_4.yaml', help='config files for VQ model')
parser.add_argument('--encoder_path', type=str, default='Pretrained/Reenactment/Obama/Enc_00161851.ckpt', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--implicit_path', type=str, default='Pretrained/Reenactment/Obama/Imp_00161851.ckpt', help='which epoch to load? set to latest to use latest cached model')

# for temporal and codebook settings
parser.add_argument('--multi_heat', type=bool, default=True, help='whether to use multiple heatmaps at one time')
parser.add_argument('--repeat_heat', type=bool, default=False, help='whether to use multiple heatmaps at one time')
parser.add_argument('--heat_num', type=int, default=7, help='whether to use multiple heatmaps at one time')
parser.add_argument('--geo_weight', type=float, default=100.0, help='weight for geometry supervision')
parser.add_argument('--use_ref', type=bool, default=True, help='whether to use multiple heatmaps at one time')
parser.add_argument('--mean_codebook', type=bool, default=True, help='whether to use mean codebook')
parser.add_argument('--pooling', type=str, default='mean', help='whether to use multiple heatmaps at one time')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

test_dataset = GridDataset(opt)
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=2)

trainer = GridFormer7_Model(opt)

if os.path.exists(opt.encoder_path):

    pretrained_state_dict = torch.load(opt.encoder_path)['encoder']
    model_state_dict = trainer.Encoder.state_dict()
    model = util.dict_unite(pretrained_state_dict, model_state_dict)
    print('loading the model from %s' % opt.encoder_path)
    trainer.Encoder.load_state_dict(model)
    trainer.Encoder.cuda()

    pretrained_state_dict = torch.load(opt.implicit_path)['implicit']
    model_state_dict = trainer.Imp.state_dict()
    model = util.dict_unite(pretrained_state_dict, model_state_dict)
    print('loading the model from %s' % opt.implicit_path)
    trainer.Imp.load_state_dict(model)
    trainer.Imp.cuda()
else:
    print('You may have used the wrong directory for pretrained models!')

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    trainer = torch.nn.DataParallel(trainer)
else:
    trainer.cuda()
    print("Let's use 1 GPU")


_, image_directory = util.prepare_sub_folder(os.path.join(opt.output_path, opt.model_name))

### Load Network Weights


print('----Begin Testing----')


for i, data in enumerate(test_loader):
    ### testing code
    feature_map = trainer.set_test_input(data)
    fake_img = trainer.inference(feature_map)
    feature_map = util.batch2seq(feature_map, opt.batch_internal, infer=True)
    fake_seq = util.batch2seq(fake_img, opt.batch_internal, infer=True)

    ### sequential generate each frame in a batch
    for j in range(len(fake_seq)):
        save_path = os.path.join(image_directory, "%04d.jpg"%(i*opt.batch_internal + j))
        util.save_torch_img(fake_img[j], save_path)

command = 'ffmpeg -r 60 -f image2 -i {folder}/%04d.jpg  -vcodec libx264  {folder}/{name}.mp4'.format(folder=image_directory, name=opt.model_name)
os.system(command)
print('The Generated Video is saved in %s !'%image_directory)
        


    

