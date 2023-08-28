import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PriorTrans(nn.Module):
    def __init__(self, *, dim=256, batch_size=4, depth=4, heads=4, dim_head=64, mlp_dim=256, dropout=0.1, pooling='max'):
        super().__init__()
        self.mlp_hair = nn.Sequential(
            nn.Linear(260, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(260, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        # self.conv_hair = nn.Sequential(
        #         nn.Conv2d(dim*2, dim, 3, 1, 1, bias=False),
        #         nn.ReLU(),
        #         nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
        #     )
        self.transformer = Transformer(dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
        self.batch_size = batch_size
        self.pooling = pooling
        #self.mask = torch.zeros([self.batch_size, 256, 256])

    def forward(self, feat, codebook):
        '''
        featuremap size:[B, 256, 16, 16]
        codebook prior size: [B, 256, 256]
        '''
        
        ######### Head Branch #########
        feat_fl = rearrange(feat, 'b c h w -> b (h w) c') # [B, 256, 256]
        feat_head = torch.cat([feat_fl, codebook], dim=-1)
        feat_head = self.mlp_head(feat_head)

        ######### Hair Branch ##########
        # feat_fl = rearrange(feat, 'b c h w -> b (h w) c') # [B, 256, 256]
        num = feat_fl.shape[1]
        if self.pooling == 'max':
            feat_top = torch.max(feat_fl, dim=1, keepdim=True)[0].repeat(1, num, 1) # [B, 256, 256]
        elif self.pooling == 'mean':
            feat_top = torch.mean(feat_fl, dim=1, keepdim=True).repeat(1, num, 1) # [B, 256, 256]
        feat_top = torch.cat([feat_top, codebook], dim=-1)
        feat_top = self.mlp_hair(feat_top) # [B, 256, 256]
        # self.mask[:, :int(num/2), :] = feat_top[:, :int(num/2), :]

        ######### Global Modeling #########
        out = feat_head + feat_top
        out = rearrange(out, '(b p) n c -> p (b n) c', p=1) # [1, B*256, 256]
        out = self.transformer(out)
        out = rearrange(out, 'p (b h w) c -> (b p) c h w', p=1, b=self.batch_size, h=16, w=16)
        return out



