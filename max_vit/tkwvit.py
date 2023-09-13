from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from .attention import LinearAttention,FullAttention,RPEAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 1),
        nn.BatchNorm2d(dim_out),
        nn.SiLU(),
        nn.Conv2d(dim_out, dim_out, 3, stride = stride, padding = 1, groups = dim_out),
        SqueezeExcitation(dim_out, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(dim_out, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# class MBConv(nn.Module):
#     def __init__(
#         self,
#         dim_in,
#         dim_out,
#         *,
#         downsample,
#         expansion_rate = 4,
#         shrinkage_rate = 0.25,
#         dropout = 0.
#     ):
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.downsample = downsample
#         hidden_dim = int(expansion_rate * dim_out)
#         stride = 2 if downsample else 1
#         self.conv1 = nn.Conv2d(dim_in, dim_out, 1)
#         self.norm1 = nn.LayerNorm(dim_out)
#         self.act1 = nn.SiLU()
#         self.conv2 = nn.Conv2d(dim_out, dim_out, 3, stride = stride, padding = 1, groups = dim_out)
#         self.se = SqueezeExcitation(dim_out, shrinkage_rate = shrinkage_rate)
#         self.conv3 = nn.Conv2d(dim_out, dim_out, 1)
#         self.norm2 = nn.LayerNorm(dim_out)

#     def forward(self,x):
#         _x = self.conv1(x)
#         _x = self.norm1(_x.permute(0,2,3,1)).permute(0,3,1,2)
#         _x = self.act1(_x)
#         _x = self.conv2(_x)
#         _x = self.se(_x)
#         _x = self.conv3(_x)
#         _x = self.norm2(_x.permute(0,2,3,1)).permute(0,3,1,2)
#         if self.dim_in == self.dim_out:
#             _x = x+_x
#         return _x

# attention related classes


class TopKWindowAttentionLayer(nn.Module):#纯粹的TopKWindowAttention
    def __init__(self,d_model,nhead,attention,w=7,k=8):
        super(TopKWindowAttentionLayer, self).__init__()
        self.w = w
        self.k = k
        self.dim = d_model // nhead
        self.nhead = nhead
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        
        if attention == "linear":#先乘kv再乘q
            self.attention = LinearAttention()
        elif attention == "full":#正常qk，qkv
            self.attention = FullAttention()
        elif attention == "rpe":#相对位置编码再attn
            self.attention = RPEAttention(d_model)
        else:
            raise NotImplementedError()
        self.merge = nn.Linear(d_model, d_model, bias=False)#普通的全连接层

        self.mlp = Mlp(in_features=2*d_model,hidden_features=2*d_model,out_features=d_model)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # self.peg = PositionEmbedding(d_model)

    def forward(self, x):
        bs,d,h,w = x.shape
        m = h // self.w#高能切多少个窗子
        n = w // self.w#宽能切多少个窗子
        x = rearrange(x,'b d h w -> b (h w) d')#变成一条东西
        
        query, key, value = x, x, x
        # TopK-Window-multihead-Attention
        queries = self.q_proj(query)
        keys = self.k_proj(key)
        values = self.v_proj(value)

        queries = rearrange(queries,'b (m w1 n w2) d -> b (m n) (w1 w2) d',m=m,w1=self.w,n=n,w2=self.w)#这是把sequence还原回切好window的时候了
        keys = rearrange(keys,'b (m w1 n w2) d -> b (m n) (w1 w2) d',m=m,w1=self.w,n=n,w2=self.w)#
        values = rearrange(values,'b (m w1 n w2) d -> b (m n) (w1 w2) d',m=m,w1=self.w,n=n,w2=self.w)
        query_mean = torch.mean(queries,dim=2)# linear要删，开始  注:linear和full 在359行设置
        key_mean = torch.mean(keys,dim=2)
        value_mean = torch.mean(values,dim=2)
          
        window_similarity = torch.einsum('bmd,bnd->bmn',query_mean,key_mean)#就是(b,mn,mn)
        topk_values,topk_indices = torch.topk(window_similarity,dim=-1,k=self.k)#给每个Q窗子找出它最亲近的k个K窗子 (b,mn,k)
        
        fine_keys = []
        fine_values = []
        for i in range(bs):
            fine_keys.append(keys[i][topk_indices[i]])#给每个patch都记录下与他最匹配的k个patch的key。len(fine_key)=b  写成tensor就是([b,mn, k, window面积, d])
            fine_values.append(values[i][topk_indices[i]])#给每个patch都记录下与他最匹配的k个patch的value。
            
        fine_keys = torch.stack(fine_keys).reshape(bs,m*n,-1,d) # [B, m*n, k*w1*w2, D] 将fine_keys压成sequence
        fine_values = torch.stack(fine_values).reshape(bs,m*n,-1,d)
        
        keys = torch.cat([fine_keys,torch.tile(key_mean.unsqueeze(1),(1,m*n,1,1))],2)
        values = torch.cat([fine_values,torch.tile(value_mean.unsqueeze(1),(1,m*n,1,1))],2)#linear要删，结束
        
        queries = rearrange(queries,'b nw ws (h d) -> b (nw ws) h d',h=self.nhead)
        keys = rearrange(keys,'b nw ws (h d) -> b (nw ws) h d',h=self.nhead)
        values = rearrange(values,'b nw ws (h d) -> b (nw ws) h d',h=self.nhead)
        
        message = self.attention(queries, keys, values, q_mask=None, kv_mask=None)  # [N, L, (H, D)]
        message = torch.cat([x,self.norm1(self.merge(message.reshape(bs,-1,d)))],dim=2)
        # x = self.norm1(self.merge(message.reshape(bs,-1,d))) + x

        # feed-forward
        # x = self.norm2(self.mlp(x,h,w)) + x
        x = self.norm2(self.mlp(message,h,w)) + x
        
        x = rearrange(x,'b (h w) d -> b d h w',h=h,w=w)
        return x
    


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.dwconv(x, H, W)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class TopKWindowViT(nn.Module):#自己的模型不用四个topk stage，两个够了,要后两个

    def __init__(
        self,
        *,
        dims=[256,512],
        depths=(5,2),
        dim_head = 32,
        dim_conv_stem = None,
        window_size = [4,2],
        ks = [4,4],#topk挑几个出来
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        in_chans = 128,
    ):
        super().__init__()
        assert isinstance(depths, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem 这个不要

        # dim_conv_stem = default(dim_conv_stem, 64)
        # 
        # self.conv_stem = nn.Sequential(#原图高宽减半，通道64
        #     nn.Conv2d(in_chans, dim_conv_stem, 7, stride = 2, padding = 3),
        #     nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        # )

        # variables

        num_stages = len(depths)#(2,2,5,2)

        # dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (in_chans, *dims)#[128,256,512,1024]
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])
        self.out_norms = nn.ModuleList()
        # shorthand for window size for efficient block - grid like attention


        # iterate through stages
        # 0,((64, 128), 2)
        # 1,((128, 256), 2)
        # 2,((256, 512), 5)
        # 3,((512, 1024), 2)
        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
            self.out_norms.append(nn.LayerNorm(layer_dim))
            self.layers.append(nn.ModuleList())
            layers = self.layers[ind]
            k = ks[ind]#[8,6,4,4],#topk挑几个出来
            w = window_size[ind]#[8,8,4,2]
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                block = nn.Sequential(
                    MBConv(#mobilenet conv
                        stage_dim_in,
                        layer_dim,
                        # downsample = (is_first and ind!=0),
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    TopKWindowAttentionLayer(layer_dim, layer_dim // dim_head, attention="linear",w=w,k=k)
                )
                layers.append(block)
        
        
    def forward(self, x):
        # x = self.conv_stem(x)#原图高宽减半，通道64,改到自己的模型就不用这个了。r

        for i,stage in enumerate(self.layers):
            for _stage in stage:
                x = _stage(x)
        return x



if __name__ == "__main__":
    config = {
        # "dims":[128,192,256,320],
        "dims":[128,256,512,1024],
        "depths":(1,1,2,1)
    }
    model = TopKWindowViT(**config).cuda()
    x = torch.rand(1,3,256,256).cuda()
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(y.shape)
    
