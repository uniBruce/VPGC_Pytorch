import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from main import instantiate_from_config
from taming.models.vqgan import VQModel
from taming.modules.diffusionmodules.model import VUNet
from utils.import_utils import instantiate_from_config, get_obj_from_str
from taming.models.vqdouble import VQDoubleModel, VQDoubleMaskModel



class VQRefModel(VQDoubleMaskModel):
    def __init__(self,
                 teacher_config,
                 unetconfig,
                 lossconfig,
                 **kwargs
                 ):
        super().__init__(teacher_config,
                        unetconfig,
                        lossconfig,
                        **kwargs)


    def training_step(self, batch, batch_idx, optimizer_idx):
        ## 1. student will use the teacher's codebook and learning to inpaint on a masked image
        x, z, mask, x_mask = self.get_input(batch)
        if self.maskz:
            z = self.mask_z(z)
        ref_x, ref_z, ref_mask, ref_x_mask = self.get_input(batch, prefix='ref_')
        clip_len = batch['clip_len'][0]

        if clip_len != 1:
            ref_x = ref_x.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x.shape)
            ref_mask = ref_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(mask.shape)
            ref_x_mask = ref_x_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x_mask.shape)
            ref_z = ref_z.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(z.shape)

        cat_x = torch.cat([ref_x, x], 1)

        cat_x_mask = torch.cat([ref_x, x_mask], 1)
        # cat_z = torch.cat([ref_z, z], 1)

        xrec = self(cat_x_mask, z)

        # xrec, qloss = self(x) only for visualization
        qloss = torch.zeros(1).requires_grad_(True)
        if optimizer_idx == 0:
            # autoencode
            if self.no_continous_loss:
                B, C, H, W = cat_x.shape
                cat_x = cat_x.view(-1, C, H, W)
                xrec = xrec.view(-1, C, H, W)

            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, mask=mask, clip_len=clip_len,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            if self.no_continous_loss:
                B, C, H, W = cat_x.shape
                cat_x = cat_x.view(-1, C, H, W)
                xrec = xrec.view(-1, C, H, W)
            
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, clip_len=clip_len,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss


    def log_images(self, batch, **kwargs):
        log = dict()
        x, z, mask, x_mask = self.get_input(batch)
        ref_x, ref_z, ref_mask, ref_x_mask = self.get_input(batch, prefix='ref_')
        if self.maskz:
            z = self.mask_z(z)

        clip_len = batch['clip_len'][0]

        if clip_len != 1:
            ref_x = ref_x.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x.shape)
            ref_mask = ref_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(mask.shape)
            ref_x_mask = ref_x_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x_mask.shape)
            ref_z = ref_z.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(z.shape)
        cat_x_mask = torch.cat([ref_x, x_mask], 1)
        # x, x_mask, z = self.get_input(batch)
        # import pdb; pdb.set_trace()
        # vq_xrec, _ = self.teacher(x)
        xrec = self(cat_x_mask, z)
        # xrec_ref = xrec[:,:3,:,:]

        log["x_mask"] = x_mask
        # log["x_mask_ref"] = ref_x_mask
        # log["vq_xrec"] = vq_xrec
        log["vqpp_xrec_x"] = xrec
        # log["vqpp_xrec_ref"] = xrec_ref
        log["x"] = x
        log["x_ref"] = ref_x

        return log