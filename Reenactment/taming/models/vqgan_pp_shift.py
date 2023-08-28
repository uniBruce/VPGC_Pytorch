import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from main import instantiate_from_config
from taming.models.vqgan import VQModel
from taming.models.vqgan_pp import VQPPModel
from taming.modules.diffusionmodules.model import VUNet
from utils.import_utils import instantiate_from_config, get_obj_from_str
from taming.models.shift_utils import *



class VQPPShiftModel(VQPPModel):
    def __init__(self,
                 teacher_config,
                 lossconfig,
                 **kwargs
                 ):
        super().__init__(teacher_config,
                        lossconfig,
                        **kwargs)
    
    def forward(self, x_mask, z):
        z = z.detach().requires_grad_(True)
        xrec = self.VUNet(x_mask, z)
        return xrec

    def get_last_layer(self):
        return self.VUNet.conv_out.weight

    def get_input(self, batch):
        x = self.teacher.get_input(batch, self.image_key)
        mask = self.teacher.get_input(batch, self.mask_key)
        x, mask = [ii.to(self.device) for ii in [x, mask]]
        x_mask = (x*mask).detach().requires_grad_(True)
        
        z = self.teacher.encode_code(x)
        affine_mat = compute_affine_matrices(x.shape[0], [x.shape[2], x.shape[3]])
        
        grid = F.affine_grid(affine_mat, x.shape).to(x.device)
        x_shift = F.grid_sample(x.clone(), grid)
        ## network could not cover this case
        # x_mask_shift = F.grid_sample(x_mask.clone(), grid)
        x_mask_shift = (x_shift*mask).detach().requires_grad_(True)
        z_shift = self.teacher.encode_code(x_shift)

        return [x, x_mask, z], [x_shift, x_mask_shift, z_shift], grid

    def training_step(self, batch, batch_idx, optimizer_idx):
        ## 1. student will use the teacher's codebook and learning to inpaint on a masked image
        [x, x_mask, z], [x_shift, x_mask_shift, z_shift], grid = self.get_input(batch)
        xrec = self(x_mask, z)
        x_rec_shift = F.grid_sample(xrec.clone(), grid).detach()
        x_shift_rec = self(x_mask_shift, z_shift)

        # xrec, qloss = self(x) only for visualization
        qloss = torch.zeros(1).requires_grad_(True)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, x_rec_shift, x_shift_rec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, x_rec_shift, x_shift_rec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.VUNet.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        [x, x_mask, z], [x_shift, x_mask_shift, z_shift], grid = self.get_input(batch)
        # import pdb; pdb.set_trace()
        vq_xrec, _ = self.teacher(x)
        xrec = self(x_mask, z)

        x_rec_shift = F.grid_sample(xrec.clone(), grid).detach()
        x_shift_rec = self(x_mask_shift, z_shift)

        log["x_mask"] = x_mask
        log["vq_xrec"] = vq_xrec
        log["vqpp_xrec"] = xrec
        log["x"] = x
        log["x_mask_shift"] = x_mask_shift
        log["x_rec_shift"] = x_rec_shift
        log["x_shift_rec"] = x_shift_rec

        return log
