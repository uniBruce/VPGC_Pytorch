import torch
import importlib
import torch.nn.functional as F
import pytorch_lightning as pl

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
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


class VQSEQModel(pl.LightningModule):
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
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape, use_orthgonal=use_orthgonal)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
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

    def requant(self, x):
        quant, emb_loss, info = self.quantize(x)
        return quant

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

    # def inference(self, batch):
    #      with torch.no_grad():
    #         x = self.get_input(batch, self.image_key)
    #         x = self.seq2batch(x)
    #         x = x.cuda()
    #         quant, diff, _ = self.encode(x)
    #         dec = self.decode(quant)
    #         return dec

    def inference(self, batch):
         with torch.no_grad():
            x = self.get_input(batch, self.image_key)
            x = self.seq2batch(x)
            x = x.cuda()
            quant, diff, _ = self.encode(x)
            dec = self.decode(quant)
            return dec

    def infer_LSP(self, code):
        code = self.quant_conv(code)
        quant, _, _ = self.quantize(code)
        dec = self.decode(quant)
        return dec

    def infer_stage(self, code, decode=False):
        code = self.quant_conv(code)
        quant, _, _ = self.quantize(code)
        if decode:
            dec = self.decode(quant)
            return dec
        else:
            return quant

    def encode_to_z(self, x):
        quant_z, _, info = self.encode(x)
        indices = info[2].view(-1)
        #indices = self.permuter(indices)
        return quant_z, indices

    def encode_to_h(self, x):
        h, quant_z, _, info = self.encode_h(x)
        indices = info[2].view(-1)
        #indices = self.permuter(indices)
        return h, quant_z, indices

    def logits_to_token(self, logits, image):
        quant_z, label = self.encode_to_z(image)
        #gt_feat, quant_z, label = self.encode_to_h(image)
        probs = F.softmax(logits, dim=-1)
        shape = probs.shape
        probs = probs.reshape(shape[0]*shape[1],shape[2])
        # idx = torch.multinomial(probs, num_samples=1)
        # value, idx = torch.max(probs, dim=-1)
        _, idx = torch.topk(probs, k=1, dim=-1)
        idx = idx.reshape(shape[0],shape[1])
        return probs, idx, label, quant_z

    def logits_to_token_train(self, logits, image):
        #quant_z, label = self.encode_to_z(image)
        gt_feat, quant_z, label = self.encode_to_h(image)
        probs = F.softmax(logits, dim=-1)
        shape = probs.shape
        probs = probs.reshape(shape[0]*shape[1],shape[2])
        # idx = torch.multinomial(probs, num_samples=1)
        # value, idx = torch.max(probs, dim=-1)
        _, idx = torch.topk(probs, k=1, dim=-1)
        idx = idx.reshape(shape[0],shape[1])
        return gt_feat, probs, idx, label, quant_z

    def logits_to_token_infer(self, logits):
        probs = F.softmax(logits, dim=-1)
        shape = probs.shape
        probs = probs.reshape(shape[0]*shape[1],shape[2])
        # value, _ = torch.max(probs, dim=-1)
        _, idx = torch.topk(probs, k=1, dim=-1)
        idx = idx.reshape(shape[0],shape[1])
        return idx

    def decode_to_img(self, index, zshape):
        #index = self.permuter(index, reverse=True)
        print('index shape', index.shape)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.decode(quant_z)
        return x

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


# class VQSegmentationModel(VQModel):
#     def __init__(self, n_labels, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
#                                   list(self.decoder.parameters())+
#                                   list(self.quantize.parameters())+
#                                   list(self.quant_conv.parameters())+
#                                   list(self.post_quant_conv.parameters()),
#                                   lr=lr, betas=(0.5, 0.9))
#         return opt_ae

#     def training_step(self, batch, batch_idx):
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x)
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
#         self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#         return aeloss

#     def validation_step(self, batch, batch_idx):
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x)
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
#         self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#         total_loss = log_dict_ae["val/total_loss"]
#         self.log("val/total_loss", total_loss,
#                  prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
#         return aeloss

#     @torch.no_grad()
#     def log_images(self, batch, **kwargs):
#         log = dict()
#         x = self.get_input(batch, self.image_key)
#         x = x.to(self.device)
#         xrec, _ = self(x)
#         if x.shape[1] > 3:
#             # colorize with random projection
#             assert xrec.shape[1] > 3
#             # convert logits to indices
#             xrec = torch.argmax(xrec, dim=1, keepdim=True)
#             xrec = F.one_hot(xrec, num_classes=x.shape[1])
#             xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
#             x = self.to_rgb(x)
#             xrec = self.to_rgb(xrec)
#         log["inputs"] = x
#         log["reconstructions"] = xrec
#         return log


# class VQNoDiscModel(VQModel):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                  n_embed,
#                  embed_dim,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None
#                  ):
#         super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
#                          ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
#                          colorize_nlabels=colorize_nlabels)

#     def training_step(self, batch, batch_idx):
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x)
#         # autoencode
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
#         output = pl.TrainResult(minimize=aeloss)
#         output.log("train/aeloss", aeloss,
#                    prog_bar=True, logger=True, on_step=True, on_epoch=True)
#         output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#         return output

#     def validation_step(self, batch, batch_idx):
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x)
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
#         rec_loss = log_dict_ae["val/rec_loss"]
#         output = pl.EvalResult(checkpoint_on=rec_loss)
#         output.log("val/rec_loss", rec_loss,
#                    prog_bar=True, logger=True, on_step=True, on_epoch=True)
#         output.log("val/aeloss", aeloss,
#                    prog_bar=True, logger=True, on_step=True, on_epoch=True)
#         output.log_dict(log_dict_ae)

#         return output

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(list(self.encoder.parameters())+
#                                   list(self.decoder.parameters())+
#                                   list(self.quantize.parameters())+
#                                   list(self.quant_conv.parameters())+
#                                   list(self.post_quant_conv.parameters()),
#                                   lr=self.learning_rate, betas=(0.5, 0.9))
#         return optimizer


# class GumbelVQ(VQModel):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                  n_embed,
#                  embed_dim,
#                  temperature_scheduler_config,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None,
#                  monitor=None,
#                  kl_weight=1e-8,
#                  remap=None,
#                  ):

#         z_channels = ddconfig["z_channels"]
#         super().__init__(ddconfig,
#                          lossconfig,
#                          n_embed,
#                          embed_dim,
#                          ckpt_path=None,
#                          ignore_keys=ignore_keys,
#                          image_key=image_key,
#                          colorize_nlabels=colorize_nlabels,
#                          monitor=monitor,
#                          )

#         self.loss.n_classes = n_embed
#         self.vocab_size = n_embed

#         self.quantize = GumbelQuantize(z_channels, embed_dim,
#                                        n_embed=n_embed,
#                                        kl_weight=kl_weight, temp_init=1.0,
#                                        remap=remap)

#         self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

#         if ckpt_path is not None:
#             self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

#     def temperature_scheduling(self):
#         self.quantize.temperature = self.temperature_scheduler(self.global_step)

#     def encode_to_prequant(self, x):
#         h = self.encoder(x)
#         h = self.quant_conv(h)
#         return h

#     def decode_code(self, code_b):
#         raise NotImplementedError

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         self.temperature_scheduling()
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x)

#         if optimizer_idx == 0:
#             # autoencode
#             aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
#                                             last_layer=self.get_last_layer(), split="train")

#             self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             return aeloss

#         if optimizer_idx == 1:
#             # discriminator
#             discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
#                                             last_layer=self.get_last_layer(), split="train")
#             self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             return discloss

#     def validation_step(self, batch, batch_idx):
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss = self(x, return_pred_indices=True)
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
#                                         last_layer=self.get_last_layer(), split="val")

#         discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
#                                             last_layer=self.get_last_layer(), split="val")
#         rec_loss = log_dict_ae["val/rec_loss"]
#         self.log("val/rec_loss", rec_loss,
#                  prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
#         self.log("val/aeloss", aeloss,
#                  prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
#         self.log_dict(log_dict_ae)
#         self.log_dict(log_dict_disc)
#         return self.log_dict

#     def log_images(self, batch, **kwargs):
#         log = dict()
#         x = self.get_input(batch, self.image_key)
#         x = x.to(self.device)
#         # encode
#         h = self.encoder(x)
#         h = self.quant_conv(h)
#         quant, _, _ = self.quantize(h)
#         # decode
#         x_rec = self.decode(quant)
#         log["inputs"] = x
#         log["reconstructions"] = x_rec
#         return log


# class EMAVQ(VQModel):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                  n_embed,
#                  embed_dim,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None,
#                  monitor=None,
#                  remap=None,
#                  sane_index_shape=False,  # tell vector quantizer to return indices as bhw
#                  ):
#         super().__init__(ddconfig,
#                          lossconfig,
#                          n_embed,
#                          embed_dim,
#                          ckpt_path=None,
#                          ignore_keys=ignore_keys,
#                          image_key=image_key,
#                          colorize_nlabels=colorize_nlabels,
#                          monitor=monitor,
#                          )
#         self.quantize = EMAVectorQuantizer(n_embed=n_embed,
#                                            embedding_dim=embed_dim,
#                                            beta=0.25,
#                                            remap=remap)
#     def configure_optimizers(self):
#         lr = self.learning_rate
#         #Remove self.quantize from parameter list since it is updated via EMA
#         opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
#                                   list(self.decoder.parameters())+
#                                   list(self.quant_conv.parameters())+
#                                   list(self.post_quant_conv.parameters()),
#                                   lr=lr, betas=(0.5, 0.9))
#         opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
#                                     lr=lr, betas=(0.5, 0.9))
#         return [opt_ae, opt_disc], []                                           