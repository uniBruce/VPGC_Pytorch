import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.GANloss import GANLoss
from taming.modules.discriminator.model import NLayerDiscriminator, MultiscaleDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0, multiframe_D=False,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, mask_loss=True, rec_ref=False,
                 disc_ndf=64, disc_loss="hinge", D_type='multiscale'):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.mask_loss = mask_loss
        self.rec_ref = rec_ref
        self.multiframe_D = multiframe_D
        
        self.disc_loss = GANLoss(gan_mode=disc_loss)
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")

        assert D_type in ["multiscale", "nlayer", "seq"]
        self.D_type = D_type
        if D_type == 'multiscale':
            self.discriminator = MultiscaleDiscriminator(input_nc=disc_in_channels,
                                                    n_layers_D=disc_num_layers, 
                                                    ndf=disc_ndf, 
                                                    use_actnorm=use_actnorm
                                                    ).apply(weights_init)
        elif D_type == 'nlayer':
            self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        else:
            self.discriminator = MultiscaleDiscriminator(input_nc=disc_in_channels,
                                                    n_layers_D=disc_num_layers, 
                                                    ndf=disc_ndf, 
                                                    use_actnorm=use_actnorm
                                                    ).apply(weights_init)



        self.discriminator_iter_start = disc_start
        # if disc_loss == "hinge":
        #     self.disc_loss = hinge_d_loss
        # elif disc_loss == "vanilla":
        #     self.disc_loss = vanilla_d_loss
        # else:
        #     raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        # import pdb; pdb.set_trace()
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def batch2seq(self, input, batch_size=4, infer=False):
        tensor_list =  []
        for i in range(batch_size):
            tensor_list.append(input[i:(i+1), :, :, :])
        if not infer:
            return torch.cat(tensor_list, 1)
        else:
            return tensor_list

    def run_discriminator_one_step(self, inputs, reconstructions, global_step, cond=None, split='train'):
            
            if cond is None:
                if self.D_type == 'seq':
                    input_seq = self.batch2seq(inputs)
                    rec_seq = self.batch2seq(reconstructions)
                    logits_real = self.discriminator(input_seq.contiguous().detach())
                    logits_fake = self.discriminator(rec_seq.contiguous().detach())
                else:
                    logits_real = self.discriminator(inputs.contiguous().detach())
                    logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss_fake = self.disc_loss(logits_fake, False,
                                               for_discriminator=True)
            d_loss_real = self.disc_loss(logits_real, True,
                                               for_discriminator=True)
            d_loss = disc_factor * (d_loss_real + d_loss_fake) * 0.5

            if self.D_type == 'multiscale' or self.D_type == 'seq':
                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                    "{}/d_loss_real".format(split): d_loss_real.detach().mean(),
                    "{}/d_loss_fake".format(split): d_loss_fake.detach().mean()
                    }
            else:
                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                    "{}/logits_real".format(split): logits_real.detach().mean(),
                    "{}/logits_fake".format(split): logits_fake.detach().mean()
                    }
            return d_loss, log

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, 
                global_step, clip_len=1, mask=None, last_layer=None, cond=None, split="train"):
        B, C, H, W = inputs.shape
        if C != 3:
            inputs = inputs.view(-1, 3, H, W)
            reconstructions = reconstructions.view(-1, 3, H, W)
            if mask is not None:
                mask = mask.view(-1, 1, H, W)
        if mask is None or not self.mask_loss:
            mask = torch.ones_like(inputs).to(inputs.device)
        else:
            mask = mask * 0.9 + torch.ones_like(inputs).to(inputs.device) * 0.1
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) * mask
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()) * mask
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        if C != 3:
            p_loss = p_loss.view(B, -1, 1, 1, 1).mean(1)
            # p_loss = torch.mean(p_loss, 1)
            inputs = inputs.view(B, C, H, W)
            reconstructions = reconstructions.view(B, C, H, W)
            if self.multiframe_D:
                inputs = inputs[:,3:,:,:].contiguous()
                reconstructions = reconstructions[:,3:,:,:].contiguous()
        if self.multiframe_D and clip_len != 1:
            inputs = inputs.view(-1, 3 * clip_len, H, W)
            reconstructions = reconstructions.view(-1, 3 * clip_len, H, W)
        

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

            # g_loss = -torch.mean(logits_fake)
            g_loss = self.disc_loss(logits_fake, True, for_discriminator=False)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            return self.run_discriminator_one_step(inputs, reconstructions, global_step, cond, split)


class VQLPIPSWithSeqDiscriminator(VQLPIPSWithDiscriminator):
    def __init__(self, disc_start, **kwargs):
        super(VQLPIPSWithSeqDiscriminator, self).__init__(disc_start, **kwargs)

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                if self.D_type == 'seq':
                    rec_seq = self.batch2seq(reconstructions)
                    logits_fake = self.discriminator(rec_seq.contiguous())
                else:
                    logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            
            g_loss = self.disc_loss(logits_fake, True, for_discriminator=False)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

        
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            return self.run_discriminator_one_step(inputs, reconstructions, global_step, cond, split)


class VQLPIPSShiftWithDiscriminator(VQLPIPSWithDiscriminator):
    def __init__(self, disc_start, **kwargs):
        super(VQLPIPSShiftWithDiscriminator, self).__init__(disc_start, **kwargs)

    def forward(self, codebook_loss, inputs, reconstructions, inputs_rec_shift, inputs_shift_rec, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            
            g_loss = self.disc_loss(logits_fake, True, for_discriminator=False)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            shift_loss = torch.abs(inputs_rec_shift.contiguous() - inputs_shift_rec.contiguous())
        
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean() + shift_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/shift_loss".format(split): shift_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            return self.run_discriminator_one_step(inputs, reconstructions, global_step, cond, split)
