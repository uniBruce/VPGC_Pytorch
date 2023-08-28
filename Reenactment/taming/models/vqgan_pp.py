import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from main import instantiate_from_config
from taming.models.vqgan import VQModel
from taming.modules.diffusionmodules.model import VUNet
from utils.import_utils import instantiate_from_config, get_obj_from_str


class VQPPModel(pl.LightningModule):
    def __init__(self,
                 teacher_config,
                 lossconfig,
                 image_key="image",
                 mask_key="mask",
                 ignore_keys=[],
                 ckpt_path=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.mask_key = mask_key
        self.loss = instantiate_from_config(lossconfig)
        self.teacher = instantiate_from_config(teacher_config)
        ddconfig = teacher_config['params']['ddconfig']
        self.VUNet = VUNet(**ddconfig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor
        requires_grad(self.teacher, False)
        # import pdb; pdb.set_trace()
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        # print(path)
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
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def get_input(self, batch):
        x = self.get_single_input(batch, self.image_key)
        mask = self.get_single_input(batch, self.mask_key)
        x, mask = [ii.to(self.device) for ii in [x, mask]]
        x_mask = (x*mask).detach().requires_grad_(True)
        
        with torch.no_grad():
            z = self.teacher.encode_code(x)
        return x, x_mask, z

    def training_step(self, batch, batch_idx, optimizer_idx):
        ## 1. student will use the teacher's codebook and learning to inpaint on a masked image
        x, x_mask, z = self.get_input(batch)
        xrec = self(x_mask, z)

        # xrec, qloss = self(x) only for visualization
        qloss = torch.zeros(1).requires_grad_(True)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
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
        x, x_mask, z = self.get_input(batch)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            vq_xrec, _ = self.teacher(x)
        xrec = self(x_mask, z)

        log["x_mask"] = x_mask
        log["vq_xrec"] = vq_xrec
        log["vqpp_xrec"] = xrec
        log["x"] = x

        return log

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag