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

class CrossAttention(nn.Module):
    def __init__(self, dim=16, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.heads = heads

    def forward(self, x1, x2):
        if len(x1.shape) == 4:
            B, N, G, C = x1.shape
            x1 = x1.view(-1, G, C)
        elif len(x2.shape) == 4:
            B, N, G, C = x2.shape
            x2 = x2.view(-1, G, C)
        #print(x1.shape, x2.shape)
        f1 = self.to_q(x1)
        f2 = self.to_k(x2)
        fv = self.to_v(x2)
        # f1_head = rearrange(f1, 'b n (h d) -> b h n d', h = self.heads)
        # f2_head = rearrange(f2, 'b n (h d) -> b h n d', h = self.heads)
        # fv_head = rearrange(fv, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(f1, f2.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, fv)
        #out = rearrange(out, 'b h n d -> b n (h d)')
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         PreNorm(dim, CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
        #         PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        #     ])
        self.attn = CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout = dropout)
    def forward(self, x1, x2):
        # for attn, ff in self.layers:
        #     x = attn.forward(x1, x2) + x1
        #     x = ff(x) + x
        x = self.attn.forward(x1, x2)
        x = self.ff.forward(x) + x
        return x

class CrossViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, group=16, num_classes=1024, batch_size=4, channels = 2, hidden_list = [8, 32, 64], dim_head = 64, dropout = 0.):
        super().__init__()
        layers= []
        imp = []
        lastv = channels
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden, bias=False))
            layers.append(nn.ReLU())
            lastv = hidden

        lastv = 128
        for hidden in [128, 256, 256]:
            imp.append(nn.Linear(lastv, hidden, bias=False))
            imp.append(nn.ReLU())
            lastv = hidden

        self.project = nn.Sequential(*layers)
        self.implicit = nn.Sequential(*imp)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.group = group

        # self.mlp_head = nn.Sequential(
        #     #nn.Linear(dim*2, dim*2),
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

        ######### Position Cooedinates #########
        self.coord = torch.from_numpy(self.mesh_grid(16)).unsqueeze(0).repeat(batch_size, 1, 1) # [B, 256, 2]
        self.coord = self.coord.to(torch.float32)
        self.coord = self.coord.cuda()


    def linear_space(self, num):
        internal = 1.0 / num
        start = internal/2.0
        x_list = [start]
        for i in range(1, num):
            x_list.append(start + i*internal)
        return np.array(x_list)

    def mesh_grid(self, num):
        x = self.linear_space(num)
        y = self.linear_space(num)
        u,v = np.meshgrid(x, y) # [16, 16]
        u = np.expand_dims(u, 2)
        v = np.expand_dims(v, 2)
        uv = np.concatenate([u,v], -1)
        uv = np.reshape(uv, (-1, 2))
        return uv

    def forward(self, ldmk, codebook):
        '''
        ldmk: size[B, N, 2] Query (Irregular coordinates)
        codebook: size[B, 256, 1024, 16] Key (Irregular coordinates)
        '''

        # ldmk_feat = self.project(ldmk)
        # coord_feat = self.project(self.coord)
        # x = self.transformer(coord_feat, ldmk_feat)
        feat = self.implicit_func(ldmk, self.coord)
        feat = feat.view(-1, self.group, int(256/self.group))
        x = self.transformer(feat, codebook)
        #x = x.view(batch_size, 256, 16, 16)
        
        return x

    def implicit_func(self, ldmk, coord):
        ldmk_feat = self.project(ldmk)
        coord_feat = self.project(self.coord)
        ldmk_global = torch.max(ldmk_feat, dim=1, keepdim=True)[0].repeat(1, coord.shape[1], 1)
        feat = torch.cat([coord_feat, ldmk_global], dim=-1)
        out = self.implicit(feat)
        return out

