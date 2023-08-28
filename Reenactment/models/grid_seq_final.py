import os
import os.path
import sys
sys.path.append("..") 
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast as autocast
from util import util
from . import networks
from . import seq_prior_final
from taming.modules.diffusionmodules.model import Encoder
from taming.models.vqgan_slidinggrid import VQGridModel
from .base_model import BaseModel
from .losses import GANLoss, MaskedL1Loss, VGGLoss
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class GridFormer7_Model(nn.Module):          
    def __init__(self, opt):
        """Initialize the VitSoftmax_Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """ 
        super(GridFormer7_Model, self).__init__()
        self.Tensor = torch.cuda.FloatTensor
        self.opt = opt
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks 
        self.model_names = ['Encoder', 'Imp']
        self.optimizers = []
        if self.opt.isTrain:
            self.Encoder = Encoder(ch=64, out_ch=256, ch_mult=(1,1,2,2,4,4), num_res_blocks=2,
                 attn_resolutions=[16], in_channels=7, resolution=512, z_channels=256, double_z=False, pooling=self.opt.pooling)
            self.Encoder = networks.init_net(self.Encoder, init_type='normal', init_gain=0.02, gpu_ids=[0], useDDP=False)
            self.Imp = seq_prior_final.PriorTrans(batch_size = self.opt.batch_internal,
                                   dim = 256,
                                   depth = 4,
                                   heads = 4,
                                   mlp_dim = 256,
                                   dropout = 0.1)
            self.Imp = networks.init_net(self.Imp, init_type='normal', init_gain=0.02, gpu_ids=[0], useDDP=False)

        else:
            self.Encoder = Encoder(ch=64, out_ch=256, ch_mult=(1,1,2,2,4,4), num_res_blocks=2,
                 attn_resolutions=[16], in_channels=7, resolution=512, z_channels=256, double_z=False, pooling=self.opt.pooling)
            self.Imp = seq_prior_final.PriorTrans(batch_size = self.opt.batch_internal,
                                   dim = 256,
                                   depth = 4,
                                   heads = 4,
                                   mlp_dim = 256,
                                   dropout = 0.1)

        ### Loading Pre-trained VQ Model
        f = open(self.opt.model_base, 'r')
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
        ddconfig = config.model.params.ddconfig
        lossconfig = config.model.params.lossconfig
        self.VQ = VQGridModel(ddconfig, lossconfig, config.model.params.n_embed, config.model.params.embed_dim, ckpt_path=self.opt.load_path)
        #self.VQ = self.VQ.to(torch.device('cuda:1'))
        self.VQ.cuda()
        self.codebook = self.VQ.quantize.embedding #[256, 256, 4]
        if self.opt.mean_codebook:
            self.codebook = torch.mean(self.codebook, 1).unsqueeze(0).repeat(self.opt.batch_internal, 1, 1).cuda() #[B, 256, 256]
        else:
            self.codebook = rearrange(self.codebook, 'n (g k) c -> n g k c', k=4)
            self.codebook = torch.mean(self.codebook, 2).view(256, -1)
            self.codebook = self.codebook.unsqueeze(0).repeat(self.opt.batch_internal, 1, 1).cuda()
        print('Loading VQ Model Done! ', 'Window Codebook size is:', self.codebook.shape)

        # define only during training time
        if self.opt.isTrain:
            # define losses names
            self.loss_names_G = ['Feature', 'Color', 'Temporal']   
            # criterion
            self.criterionMaskL1 = MaskedL1Loss().cuda()
            self.criterionL1 = nn.L1Loss().cuda()
            self.criterionCross = nn.CrossEntropyLoss().cuda()
            
            # initialize optimizer G 
            if opt.TTUR:                
                beta1, beta2 = 0, 0.9
                lr = opt.lr / 2
            else:
                beta1, beta2 = opt.beta1, 0.999
                lr = opt.lr   
            self.optimizer_E = torch.optim.Adam([{'params': self.Encoder.module.parameters(),
                                                  'initial_lr': lr}], 
                                                lr=lr, 
                                                betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_E)
            self.optimizer_I = torch.optim.Adam([{'params': self.Imp.module.parameters(),
                                                  'initial_lr': opt.lr_imp}], 
                                                lr=opt.lr_imp, 
                                                betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_I)
            
            # fp16 training
            if opt.fp16:
                self.scaler = torch.cuda.amp.GradScaler()

    def set_input(self, data, data_info=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.feature_map, self.tgt_image = data['feature_map'], data['tgt_image']
        self.feature_map = util.seq2batch(self.feature_map, self.opt.batch_internal, multi=self.opt.multi_heat, heatmap=True)
        self.tgt_image = util.seq2batch(self.tgt_image, self.opt.batch_internal)

        self.feature_map = self.feature_map.cuda()
        self.tgt_image = self.tgt_image.cuda()

        return self.feature_map, self.tgt_image

    def set_test_input(self, data, data_info=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.feature_map = data['feature_map']
        self.feature_map = util.seq2batch(self.feature_map, self.opt.batch_internal, multi=self.opt.multi_heat, heatmap=True)
        self.feature_map = self.feature_map.cuda()

        return self.feature_map

    def forward(self):
        ''' forward pass for feature2Face
        '''  
        #self.feature_map = torch.cat([self.feature_map, self.cand_image], 1)
        self.feature_map = self.Encoder(self.feature_map)
        self.code = self.Imp(self.feature_map, self.codebook)
        self.fake_pred = self.VQ.infer_ldmk(self.code)

        ##### Obtain GT code for Input Landmark
        self.tgt_code = self.VQ.encoder(self.tgt_image)

    def compute_temporal_loss(self, code):
        temporal_loss = 0
        num_frames = code.shape[0]
        for i in range(1, num_frames):
            temporal_loss += self.criterionL1(code[i, :, :, :], code[i-1, :, :, :])
        return temporal_loss / (num_frames-1) * 10
            
    def backward_G(self):
        """Calculate GAN and other loss for the generator"""

        # L1, vgg, style loss
        loss_feature = self.criterionL1(self.code, self.tgt_code)*10
        loss_temporal = self.compute_temporal_loss(self.code)
        loss_l1 = self.criterionL1(self.fake_pred, self.tgt_image) * self.opt.lambda_L1
        
        self.loss_G = loss_feature + loss_temporal
        self.optimizer_E.zero_grad()        # set G's gradients to zero
        self.optimizer_I.zero_grad()        # set G's gradients to zero
        self.loss_G.backward()
        self.loss_dict = dict(zip(self.loss_names_G, [loss_feature, loss_l1, loss_temporal]))

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""

        self.forward()
        self.backward_G()                   # calculate graidents for G
        self.optimizer_E.step()             # udpate E's weights
        self.optimizer_I.step()             # udpate E's weights

    def inference(self, feature_map):
        """ inference process """
        print('inference process!')
        with torch.no_grad():
            feature_map = self.Encoder(feature_map)
            code = self.Imp(feature_map, self.codebook)
            fake_pred = self.VQ.infer_ldmk(code)
            return fake_pred
    
    def save(self, save_dir, epoch):
        '''Save generators, discriminators, and optimizers.'''

        model_enc_name = os.path.join(save_dir, 'Enc_%08d.ckpt' % (epoch + 1))
        model_imp_name = os.path.join(save_dir, 'Imp_%08d.ckpt' % (epoch + 1))

        opt_enc_name = os.path.join(save_dir, 'optimizer_enc.ckpt')
        opt_imp_name = os.path.join(save_dir, 'optimizer_imp.ckpt')

        torch.save({'encoder':self.Encoder.state_dict()}, model_enc_name)
        torch.save({'implicit':self.Imp.state_dict()}, model_imp_name)

        torch.save({'enc_opt': self.optimizer_E.state_dict()}, opt_enc_name)
        torch.save({'imp_opt': self.optimizer_I.state_dict()}, opt_imp_name)

    def resume(self, checkpoint_dir):
        # Load Generator
        last_model_name = util.get_model_list(checkpoint_dir, "Gan")
        state_dict = torch.load(last_model_name)
        self.classifier.load_state_dict(state_dict['gan'])

        iter = int(last_model_name[-13:-5])

        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer_gan.ckpt'))
        self.optimizer_G.load_state_dict(state_dict['gan_opt'])

        # Reinitilize schedulers
        self.model_gan_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opt.lr_decay_iters,
                                        gamma=self.opt.lr_decay_gamma, last_epoch=iter)
        print('Resume from Iteration %08d' % iter)
        return iter