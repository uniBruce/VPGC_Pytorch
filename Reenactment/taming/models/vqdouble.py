import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from main import instantiate_from_config
from taming.models.vqgan import VQModel
from taming.modules.diffusionmodules.model import VUNet
from utils.import_utils import instantiate_from_config, get_obj_from_str


class VQDoubleModel(pl.LightningModule):
    def __init__(self,
                 teacher_config,
                 unetconfig,
                 lossconfig,
                 image_key="image",
                 mask_key="mask",
                 ignore_keys=[],
                 ckpt_path=None,
                 monitor=None,
                 no_continous_loss=False
                 ):
        super().__init__()
        self.image_key = image_key
        self.mask_key = mask_key
        self.loss = instantiate_from_config(lossconfig)
        self.teacher = instantiate_from_config(teacher_config)
        self.VUNet = VUNet(**unetconfig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor
        requires_grad(self.teacher, False)
        self.no_continous_loss = no_continous_loss
        # import pdb; pdb.set_trace()
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def forward(self, x_mask, z):
        z = z.detach().requires_grad_(True)
        xrec = self.VUNet(x_mask, z)
        return xrec

    def get_last_layer(self):
        return self.VUNet.conv_out.weight

    def get_single_input(self, batch, k):
        x = batch[k]

        if x.shape[1] == batch['clip_len'][0]:
            if len(x.shape) == 4:
                x = x[..., None]
            B, clip_len, H, W, C = x.shape
            x = x.view(-1, H, W, C)
        else:
            if len(x.shape) == 3:
                x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def get_input(self, batch, prefix='', load_mask=True):
        image_key = prefix + self.image_key
        x = self.get_single_input(batch, image_key)
        x = x.to(self.device)

        if load_mask:
            mask_key = prefix + self.mask_key
            mask = self.get_single_input(batch, mask_key)
            mask = mask.to(self.device)
        else:
            mask = torch.ones_like(x).to(self.device)
        
        x_mask = (x * mask).detach().requires_grad_(True)
        with torch.no_grad():
            z = self.teacher.encode_code(x)
        return x, z, mask, x_mask

    def mask_z(self, z):
        z_mask = torch.zeros_like(z)
        ranint = torch.randint(-1, 1, (4,))
        z_mask[:, :, 9 + ranint[0]:min(16 + ranint[1], 16), 1 + ranint[2]:-2 + ranint[3]] = 1
        z_mask = z_mask.to(self.device)
        masked_z = z_mask * z
        return masked_z

    def training_step(self, batch, batch_idx, optimizer_idx):
        ## 1. student will use the teacher's codebook and learning to inpaint on a masked image
        x, z, mask, x_mask = self.get_input(batch)

        
        # z = self.mask_z(z)
        ref_x, ref_z, ref_mask, ref_x_mask = self.get_input(batch, prefix='ref_')
        clip_len = batch['clip_len'][0]

        if clip_len != 1:
            ref_x = ref_x.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x.shape)
            ref_mask = ref_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(mask.shape)
            ref_x_mask = ref_x_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x_mask.shape)
            ref_z = ref_z.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(z.shape)

        cat_x = torch.cat([ref_x, x], 1)

        cat_x_mask = torch.cat([ref_x_mask, x_mask], 1)
        cat_z = torch.cat([ref_z, z], 1)

        xrec = self(cat_x_mask, cat_z)

        # xrec, qloss = self(x) only for visualization
        qloss = torch.zeros(1).requires_grad_(True)
        if optimizer_idx == 0:
            # autoencode
            if self.no_continous_loss:
                B, C, H, W = cat_x.shape
                cat_x = cat_x.view(-1, C, H, W)
                xrec = xrec.view(-1, C, H, W)
            aeloss, log_dict_ae = self.loss(qloss, cat_x, xrec, optimizer_idx, self.global_step, mask=torch.cat([ref_mask, mask], 0), clip_len=clip_len,
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
            
            discloss, log_dict_disc = self.loss(qloss, cat_x, xrec, optimizer_idx, self.global_step, clip_len=clip_len,
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

    def log_images(self, batch, **kwargs):
        log = dict()
        x, x_mask, z, mask = self.get_input(batch)
        ref_x, ref_x_mask, ref_z, ref_mask = self.get_input(batch, is_ref=True)

        clip_len = batch['clip_len'][0]

        if clip_len != 1:
            ref_x = ref_x.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x.shape)
            ref_mask = ref_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(mask.shape)
            ref_x_mask = ref_x_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x_mask.shape)
            ref_z = ref_z.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(z.shape)
        cat_x_mask = torch.cat([ref_x_mask, x_mask], 1)
        cat_z = torch.cat([ref_z, z], 1)
        # x, x_mask, z = self.get_input(batch)
        # import pdb; pdb.set_trace()
        vq_xrec, _ = self.teacher(x)
        xrec = self(cat_x_mask, cat_z)
        xrec_x = xrec[:,3:,:,:]
        xrec_ref = xrec[:,:3,:,:]

        log["x_mask"] = x_mask
        log["x_mask_ref"] = ref_x_mask
        log["vq_xrec"] = vq_xrec
        log["vqpp_xrec_x"] = xrec_x
        log["vqpp_xrec_ref"] = xrec_ref
        log["x"] = x
        log["x_ref"] = ref_x

        return log


class VQDoubleMaskModel(VQDoubleModel):
    def __init__(self,
                 teacher_config,
                 unetconfig,
                 lossconfig,
                 maskz=True,
                 **kwargs
                 ):
        super().__init__(teacher_config,
                        unetconfig,
                        lossconfig,
                        **kwargs)
        self.maskz = maskz

    def training_step(self, batch, batch_idx, optimizer_idx):
        ## 1. student will use the teacher's codebook and learning to inpaint on a masked image
                ## 1. student will use the teacher's codebook and learning to inpaint on a masked image
        x, z, mask, x_mask = self.get_input(batch)
        # z = self.mask_z(z)
        ref_x, ref_z, ref_mask, ref_x_mask = self.get_input(batch, prefix='ref_')

        aug_x, aug_z, _, _ = self.get_input(batch, prefix='aug_', load_mask=False)

        # spectrograms = batch['spectrograms']

        if self.maskz:
            z = self.mask_z(z)
        clip_len = batch['clip_len'][0]

        if clip_len != 1:
            ref_x = ref_x.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x.shape)
            ref_mask = ref_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(mask.shape)
            ref_x_mask = ref_x_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x_mask.shape)
            ref_z = ref_z.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(z.shape)

        cat_x = torch.cat([ref_x, x], 1)

        cat_x_mask = torch.cat([ref_x, x_mask], 1)
        cat_z = torch.cat([ref_z, z], 1)

        xrec = self(cat_x_mask, cat_z)

        # xrec, qloss = self(x) only for visualization
        qloss = torch.zeros(1).requires_grad_(True)
        if optimizer_idx == 0:
            # autoencode
            if self.no_continous_loss:
                B, C, H, W = cat_x.shape
                cat_x = cat_x.view(-1, C, H, W)
                xrec = xrec.view(-1, C, H, W)
            aeloss, log_dict_ae = self.loss(qloss, cat_x, xrec, optimizer_idx, self.global_step, mask=torch.cat([ref_mask, mask], 0), clip_len=clip_len,
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
            
            discloss, log_dict_disc = self.loss(qloss, cat_x, xrec, optimizer_idx, self.global_step, clip_len=clip_len,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
    
    def log_images(self, batch, **kwargs):
        log = dict()
        x, z, mask, x_mask = self.get_input(batch)
        ref_x, ref_z, ref_mask, ref_x_mask = self.get_input(batch, prefix='ref_')

        aug_x, aug_z, _, _ = self.get_input(batch, prefix='aug_', load_mask=False)
        if self.maskz:
            z = self.mask_z(z)

        clip_len = batch['clip_len'][0]

        if clip_len != 1:
            ref_x = ref_x.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x.shape)
            ref_mask = ref_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(mask.shape)
            ref_x_mask = ref_x_mask.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(x_mask.shape)
            ref_z = ref_z.unsqueeze(1).repeat(1, clip_len, 1, 1, 1).view(z.shape)
        cat_x_mask = torch.cat([ref_x, x_mask], 1)
        cat_z = torch.cat([ref_z, z], 1)
        # x, x_mask, z = self.get_input(batch)
        # import pdb; pdb.set_trace()
        vq_xrec, _ = self.teacher(x)
        xrec = self(cat_x_mask, cat_z)
        xrec_x = xrec[:,3:,:,:]
        # xrec_ref = xrec[:,:3,:,:]

        log["x_mask"] = x_mask
        # log["x_mask_ref"] = ref_x_mask
        log["vq_xrec"] = vq_xrec
        log["vqpp_xrec_x"] = xrec_x
        # log["vqpp_xrec_ref"] = xrec_ref
        log["x"] = x
        log["aug_x"] = aug_x
        log["x_ref"] = ref_x

        return log


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag