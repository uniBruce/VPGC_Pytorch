import torch
import importlib
import torch.nn.functional as F
import pytorch_lightning as pl

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import GridQuantizer as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class VQGridModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_orthgonal=False
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=remap, sane_index_shape=sane_index_shape, use_orthgonal=use_orthgonal)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], ddconfig["z_channels"], 1)
        self.post_quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            #self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            codebook = self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            print('Codebook size:', codebook.shape)
            self.quantize = VectorQuantizer(n_embed, embed_dim, embedding=codebook, beta=0.25, remap=remap, sane_index_shape=sane_index_shape, use_orthgonal=use_orthgonal)

        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

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
        return sd['quantize.embedding']

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def encode_h(self, x):
        h = self.encoder(x)
        h_q = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h_q)
        return h, quant, emb_loss, info

    def encode_to_h(self, x):
        h, quant_z, _, info = self.encode_h(x)
        indices = info[2].view(-1)
        #indices = self.permuter(indices)
        return h, quant_z, indices

    @torch.no_grad()
    def infer_ldmk(self, h):
        h_q = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h_q)
        out = self.decode(quant)
        return out

    @torch.no_grad()
    def infer_weight(self, gt=None, weight=None):
        if gt is not None:
            h = self.encoder(gt)
            h = self.quant_conv(h)
            code, gt_weight = self.quantize.infer(h)
            return gt_weight
        else:
            code, _ = self.quantize.infer(weight=weight)
            out = self.decode(code)
            return out

    @torch.no_grad()
    def encode_code(self, x):
        quant, _, _ = self.encode(x)
        quant = self.post_quant_conv(quant)
        return quant

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def inference(self, batch):
         with torch.no_grad():
            # x = self.get_input(batch, self.image_key)
            # x = self.seq2batch(x)
            # x = x.cuda()
            x = self.get_input(batch, self.image_key)
            x = self.seq2batch(x)
            x = x.to(self.device)
            xrec, _ = self(x)
            return xrec

    def get_input(self, batch, k):
        x = batch[k]
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x

    def seq2batch(self, input, batch_size=4):
        inter = 3
        tensor_list =  []
        for i in range(batch_size):
            tensor_list.append(input[:, i*inter:(i+1)*inter, :, :])
        return torch.cat(tensor_list, 0)

    def batch2seq(self, input, batch_size=4, infer=False):
        tensor_list =  []
        for i in range(batch_size):
            tensor_list.append(input[i:(i+1), :, :, :])
        if not infer:
            return torch.cat(tensor_list, 1)
        else:
            return tensor_list

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        x = self.seq2batch(x)
        xrec, qloss = self(x)

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

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        x = self.seq2batch(x)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        # a = [item for item in list(self.quantize.parameters()) if item.device == 'cpu']
        # print(a)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = self.seq2batch(x)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x