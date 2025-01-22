'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings

warnings.filterwarnings("ignore")

from blip_models.vit import VisionTransformer, interpolate_pos_embed
from blip_models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import  BertTokenizer
from timm.models.vision_transformer import Attention as TemporalAttention
from timm.layers import Mlp, DropPath, to_2tuple
from timm.layers import PatchEmbed, Mlp, DropPath, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     if drop_prob == 0. or not training:  # drop_prob废弃率=0，或者不是训练的时候，就保持原来不变
#         return x
#     keep_prob = 1 - drop_prob  # 保持率
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (b, 1, 1, 1) 元组  ndim 表示几维，图像为4维
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 0-1之间的均匀分布[2,1,1,1]
#     random_tensor.floor_()  # 下取整从而确定保存哪些样本 总共有batch个数
#     output = x.div(keep_prob) * random_tensor  # 除以 keep_prob 是为了让训练和测试时的期望保持一致
#     # 如果keep，则特征值除以 keep_prob；如果drop，则特征值为0
#     return output  # 与x的shape保持不变

#
# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.training=True
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
# # class Attention(nn.Module):
# #     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
# #         super().__init__()
# #         self.num_heads = num_heads
# #         # q,k,v向量长度
# #         head_dim = dim // num_heads
# #         self.scale = head_dim ** -0.5
# #         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
# #         self.attn_drop = nn.Dropout(attn_drop)
# #         self.proj = nn.Linear(dim, dim)
# #         self.proj_drop = nn.Dropout(proj_drop)
# #
# #     def forward(self, x):
# #         # 这里C对应上面的E，向量的长度
# #         B, N, C = x.shape
# #         # (B, N, C) -> (3，B，num_heads, N, C//num_heads), //是向下取整的意思。
# #         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
# #         # 将qkv在0维度上切成三个数据块，q,k,v:(B，num_heads, N, C//num_heads)
# #         # 这里的效果是从每个向量产生三个向量，分别是query，key和value
# #         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
# #         # @矩阵相乘获得score (B,num_heads,N,N)
# #         attn = (q @ k.transpose(-2, -1)) * self.scale
# #         attn = attn.softmax(dim=-1)
# #         attn = self.attn_drop(attn)
# #         # (B,num_heads,N,N)@(B,num_heads,N,C//num_heads)->(B,num_heads,N,C//num_heads)
# #         # (B,num_heads,N,C//num_heads) ->(B,N,num_heads,C//num_heads)
# #         # (B,N,num_heads,C//num_heads) -> (B, N, C)
# #         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
# #         # (B, N, C) -> (B, N, C)
# #         x = self.proj(x)
# #         x = self.proj_drop(x)
# #         return x
#
# class MLP(torch.nn.Module):
#
#     def __init__(self,in_features, hidden_features, act_layer, drop=0.0):
#         super(MLP, self).__init__()
#
#         self.linear1 = nn.Linear(in_features, hidden_features)
#         self.relu = act_layer()
#         self.linear2 = nn.Linear(hidden_features, hidden_features)  # 2个隐层
#         self.relu2 = act_layer()
#         self.linear3 = nn.Linear(hidden_features, in_features)
#         self.drop_path = DropPath(drop)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         x = self.relu2(x)
#         x = self.linear3(x)
#         x = self.drop_path(x)
#         return x

class MyAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            step:int=1,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.step=step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, self.head_dim).permute(3, 1, 0, 4, 2, 5)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        k=torch.cat((k[:self.step,...],k),dim=0)[:int(-1*self.step),...]
        v=torch.cat((v[:self.step,...],v),dim=0)[:int(-1*self.step),...]
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            return attn
        #     x = attn @ v
        #
        # x = x.transpose(1, 2).reshape(B,T, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x


from einops import rearrange

class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None,type="A"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = TemporalAttention(dim, num_heads,attn_drop=attn_drop,proj_drop=drop)
        # elif ws == 1:
        #     self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        # else:
        #     self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.temporal_attn_1=MyAttention(dim, num_heads,attn_drop=attn_drop,proj_drop=drop,step=1)
        self.temporal_attn_2 = MyAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop,step=2)
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size=3,stride=1, padding=1)
        self.type=type
        self.gelu=nn.GELU()

    def forward(self, x,B):
        # x: (B*T, h*w, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # spatial
        if self.type=="A":
            temp = self.mlp(self.norm2(x))

            temp=rearrange(temp,'(b t) l c -> b t l c', b=B)

            # step_1=self.drop_path(self.temporal_attn_1(temp))
            # step_2=self.drop_path(self.temporal_attn_2(temp))
            # step=torch.cat((step_2,step_1),dim=1)
            # temp=torch.cat((step,temp),dim=1)
            # temporal
            temp = rearrange(temp, 'b t l c -> (b l) c t', b=B)

            temp = self.temporal_conv(temp)
            temp = rearrange(temp, '(b l) c t -> (b t) l c', b=B)

            # output
            x = x + self.drop_path(temp)
        elif self.type=="B":
            spatial = self.mlp(self.norm2(x))
            temp=rearrange(spatial,'(b t) l c->(b l) c t',b=B)
            temp = self.temporal_conv(temp)
            temp = rearrange(temp, '(b l) c t -> (b t) l c', b=B)
            x=x+self.gelu(temp)+self.gelu(spatial)

        #x=rearrange(x,'(b t) l c -> b t l c',b=B).mean(1)
        return rearrange(x,'(b t) l c -> b t l c',b=B).mean(1),rearrange(x,'(b t) l c -> b t l c',b=B)

class ATTNBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = TemporalAttention(dim, num_heads,attn_drop=attn_drop,proj_drop=drop)
        # elif ws == 1:
        #     self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        # else:
        #     self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.temporal_attn_1=MyAttention(dim, num_heads,attn_drop=attn_drop,proj_drop=drop,step=1)
        self.temporal_attn_2 = MyAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop,step=2)
        #self.spatial_conv = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=1)
        #self.temporal_conv = nn.Conv1d(38809, 38809, kernel_size=8, stride=8, padding=0)
        self.out_mlp=Mlp(in_features=128,hidden_features=256,out_features=128)

    def forward(self, x,B):
        # x: (B*T, h*w, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        temp = self.mlp(self.norm2(x))

        temp=rearrange(temp,'(b t) l c -> b t l c', b=B)

        step_1=self.temporal_attn_1(temp)
        step_2=self.temporal_attn_2(temp)
        step=step_1-step_2
        step = rearrange(step, 't b hed h w -> b (t hed) h w')
        step=rearrange(step,'b t h w -> b (h w) t')
        #step=self.temporal_conv(step)
        step=self.out_mlp(step)
        return rearrange(step,'b (h w) t ->b t h w',h=197).unsqueeze(2)
        #step = rearrange(step, 'b (h w) t -> b h w t', h=197).mean(dim=-1)
        # eye=torch.eye(step.shape[1])
        # eye=eye.repeat(step.shape[0],1,1).cuda()
        # temp=torch.cat((step,temp),dim=1)
        # temporal
        # temp = rearrange(temp, 'b t l c -> (b l) c t', b=B)
        #
        # temp = self.temporal_conv(temp)
        # temp = rearrange(temp, '(b l) c t -> (b t) l c', b=B)
        #
        # # output
        # x = x + self.drop_path(temp)
        # x=rearrange(x,'(b t) l c -> b t l c',b=B).mean(1)
        #return ((step)**2).sum(dim=-1).sum(dim=-1)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, query_dim, kv_dim, num_heads, output_dim=None):
        super(MultiHeadCrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.output_dim = output_dim if output_dim else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, query, key, value, mask=None, return_attn=False):
        batch_size = query.size(0)

        # Linear projections
        q = self.q_proj(query) # NLC
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape and transpose for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)


        # Combine heads
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out_proj(context)
        if return_attn:
            return output, attn
        return output
class AttentionPool3d(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim
        )
        self.num_heads = num_heads

    def forward(self, x, return_attn=False): # x: BCLHW
        # import pdb;pdb.set_trace()
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # BFC(HW) -> (HW)(BF)C
        x_mean = x.mean(dim=0,keepdim=True) # (1)(BF)C
        x = torch.cat([x_mean, x], dim=0)  # (LHW+1)BC
        x = x.permute(1,0,2).contiguous() # B(LHW+1)C
        x_mean = x_mean.permute(1,0,2).contiguous() # B(1)C

        if return_attn:
            x, attn = self.cross_attn(query=x_mean,key=x,value=x,return_attn=True) # B(1)C
            return x.squeeze(dim=-1), attn
        x = self.cross_attn(query=x_mean,key=x,value=x).squeeze(dim=1) # BC
        batch, channels = x.shape
        # x = x.view(batch,channels,1,1,1)

        return x

class TextAttentionPool3d(nn.Module):
    def __init__(self, embed_dim: int, txt_dim:int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=txt_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim
        )
        self.num_heads = num_heads

    def forward(self, x, txt_feat):
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # BC(LHW) -> (LHW)BC
        x_mean = x.mean(dim=0,keepdim=True) # (1)BC
        x = torch.cat([x_mean, x], dim=0)  # (LHW+1)BC
        x = x.permute(1,0,2).contiguous() # B(LHW+1)C
        x_mean = x_mean.permute(1,0,2).contiguous() # B(1)C

        txt_feat = txt_feat.unsqueeze(dim=1)  # BC -> B(1)C

        x = self.cross_attn(query=txt_feat,key=x,value=x) # B(1)C
        x = x.squeeze(dim=1)
        batch, channels = x.shape
        x = x.view(batch,channels,1,1,1)
        return x
class PostBlock(nn.Module):
    def __init__(
            self,t_dim,v_dim,bs=8,f=16):
        super().__init__()
        self.t_weight=nn.Linear(t_dim,t_dim,bias=True)
        self.v_weight=nn.Linear(t_dim,t_dim,bias=True)
        self.v_conv=nn.Conv1d(v_dim,t_dim,1,1)
        self.t_conv=nn.Conv1d(t_dim,t_dim,1,1)
        self.logit_scale=nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.t_fc=nn.Conv1d(t_dim,t_dim//2,1,1)
        self.v_fc=nn.Conv1d(t_dim,t_dim//2,1,1)
        self.last_t_fc=nn.Conv1d(t_dim//2,1,1,1)
        self.last_v_fc=nn.Conv1d(t_dim//2,1,1,1)
        self.relu=nn.ReLU()
        self.v_downsample=nn.AdaptiveAvgPool3d((f,1,v_dim))
        self.attn_pool=AttentionPool3d(embed_dim=v_dim,num_heads=16,output_dim=v_dim)
        self.text_attn=MultiHeadCrossAttention(
            embed_dim=t_dim,
            query_dim=t_dim,
            kv_dim=t_dim,
            num_heads=16,
            output_dim=t_dim
        )
    def global_similarity(self,Et,Ev):
        Et = Et[:, 0, :].unsqueeze(1)  # (bs,1,c1)
        Ev = Ev.mean(dim=1).unsqueeze(1)
        Et = self.t_conv(Et.permute(0, 2, 1)).permute(0, 2, 1)
        Et = self.t_weight(Et) * Et
        Ev = self.v_conv(Ev.permute(0, 2, 1)).permute(0, 2, 1)
        Ev = self.v_weight(Ev) * Ev
        visual_output = Ev
        # visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = Et
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.permute(0, 2, 1))
        return retrieve_logits

    def global_similarity_v2(self,Et,Ev):
        Et = Et[:, 0, :].unsqueeze(1)  # (bs,1,c1)
        Ev = Ev.mean(dim=1).unsqueeze(1)
        Et = self.t_conv(Et.permute(0, 2, 1)).permute(0, 2, 1)
        Et = self.t_weight(Et) * Et
        Ev = self.v_conv(Ev.permute(0, 2, 1)).permute(0, 2, 1)
        Ev = self.v_weight(Ev) * Ev
        visual_output = Ev
        # visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = Et
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.permute(0, 2, 1))
        return retrieve_logits


    def local_similarity(self,Et,Ev):#all words, frames
        batchsize=Ev.shape[0]
        Ev=Ev[:,:,1:,:]
        Et = Et[:, 1:, :]  # bs,l,c2
        Ev=rearrange(Ev,"b f (w h) c -> (b f) c w h", w=14)
        Ev=self.attn_pool(Ev)
        Ev=rearrange(Ev,"(b f) c ->b f c",b=batchsize)
        #Et=self.text_attn_pool(Et.transpose(-1,-2))
        #Ev = self.v_downsample(Ev) # bs,f,wh,c1->bs,f,1,c1
        Et = self.t_conv(Et.permute(0, 2, 1)).permute(0, 2, 1)
        Et = self.t_weight(Et) * Et
        Ev = self.v_conv(Ev.permute(0, 2, 1)).permute(0, 2, 1)
        Ev = self.v_weight(Ev) * Ev
        visual_output = Ev
        # visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = Et
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.permute(0, 2, 1))
        v_score = retrieve_logits.max(dim=-2)[0]
        t_score = retrieve_logits.max(dim=-1)[0]
        v_score = (v_score * (self.last_v_fc(self.v_fc(visual_output.permute(0, 2, 1)))).squeeze(1)).sum(dim=-1)
        t_score = (t_score * (self.last_t_fc(self.t_fc(sequence_output.permute(0, 2, 1)))).squeeze(1)).sum(dim=-1)
        return (v_score + t_score).reshape(-1, 1, 1)#

    def text_local_similarity(self,Et,Ev):# keywords, patch
        Ev = Ev.mean(dim=2)  # bs,wh,c1
        Et = Et[:, 1:, :]  # bs,l,c2
        Et = self.t_conv(Et.permute(0, 2, 1)).permute(0, 2, 1)
        Et = self.relu(self.t_weight(Et).mean(dim=2)) * Et
        Ev = self.v_conv(Ev.permute(0, 2, 1)).permute(0, 2, 1)
        Ev = self.v_weight(Ev) * Ev
        visual_output = Ev
        # visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = Et
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.permute(0, 2, 1))
        v_score = retrieve_logits.sum(dim=-2)
        t_score = retrieve_logits.sum(dim=-1)
        v_score = (v_score * (self.v_fc(visual_output.permute(0, 2, 1)).permute(0, 2, 1).squeeze(-1))).sum(dim=-1)
        t_score = (t_score * (self.t_fc(sequence_output.permute(0, 2, 1)).permute(0, 2, 1).squeeze(-1))).sum(dim=-1)
        return (v_score + t_score).reshape(-1, 1, 1)

    def forward(self,Et,Ev=None,Ef=None,type="meanPv2"):
        #Et:(bs,l,c1)
        #Ev:(bs,wh,c2)
        if type=="meanP":
            # Et=Et.permute(0,2,1)
            # Ev=Ev.permute(0,2,1)
            Et=self.t_conv(Et.permute(0,2,1)).permute(0,2,1)
            Et=self.t_weight(Et)*Et
            Ev=self.v_conv(Ev.permute(0,2,1)).permute(0,2,1)
            Ev=self.v_weight(Ev)*Ev
            visual_output = Ev
            #visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
            visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

            sequence_output = Et
            sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.permute(0,2,1))
            return retrieve_logits

        elif type=="meanPv2":
            return self.global_similarity_v2(Et,Ev)
        elif type=="fine_grained":
            return self.local_similarity(Et,Ef)
        elif type=="mix":
            Sg=self.global_similarity(Et,Ev)
            Sl=self.local_similarity(Et,Ef)
            return Sg+Sl
        elif type=="local":
            batchsize=Ef.shape[0]
            Ef=Ef[:,:,1:,:]
            Ef = rearrange(Ef, "b f l c -> (b l) c f 1")
            Ef = self.attn_pool(Ef)
            Ef = rearrange(Ef, "(b l) c ->b l c", b=batchsize)
            Ef = self.v_conv(Ef.permute(0, 2, 1)).permute(0, 2, 1)
            visual_output = self.v_weight(Ef) * Ef
            res=[]
            for idx,sequence_output in enumerate(Et[0]):
                visual_output=Ef[idx,...]
                sequence_output=self.text_attn(sequence_output.unsqueeze(0),Et[1][idx,...].unsqueeze(0),Et[1][idx,...].unsqueeze(0)).squeeze(0)
                sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
                visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.permute(1,0))
                v_score = retrieve_logits.sum(dim=-2)
                t_score = retrieve_logits.sum(dim=-1)
                v_score = (v_score * (self.last_v_fc(self.v_fc(visual_output.permute(1,0)))).squeeze(1)).sum(dim=-1)
                t_score = (t_score * (self.last_t_fc(self.t_fc(sequence_output.permute(1,0)))).squeeze(1)).sum(
                    dim=-1)
                res.append((v_score + t_score).reshape(1,-1, 1))
            Sl=torch.cat(res,dim=0)

            return Sl





class MyBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = TemporalAttention(dim, num_heads,attn_drop=attn_drop,proj_drop=drop)
        # elif ws == 1:
        #     self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        # else:
        #     self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x,B):
        # x: (B*T, h*w, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # spatial
        temp = self.mlp(self.norm2(x))
        #

        # temporal
        temp = rearrange(temp, '(b t) l c -> (b l) c t', b=B)
        temp = self.temporal_conv(temp)
        temp = rearrange(temp, '(b l) c t -> (b t) l c', b=B)

        # output
        x = x + self.drop_path(temp)
        x=rearrange(x,'(b t) l c -> b t l c',b=B).mean(1)
        return x

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def build_words_list(phrases):
    res={}
    idx=0
    for phrase in phrases:
        token_list=whitespace_tokenize(phrase)
        if token_list==[]:
            continue
        re_token_list=token_list.copy()
        re_token_list.append(idx)
        del re_token_list[0]
        for front,next in zip(token_list,re_token_list):
            res[front]=next
        idx+=1
        del re_token_list,token_list
    return res

import spacy
import pytextrank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
class BLIP_Base(nn.Module):
    def __init__(self,
                 med_config='BLIP_configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 drop_path=0.2,
                 in_chans=1024,
                 embed_dim=1024,
                 patch_size=2,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        self.temporal_block=Block(dim=1024,num_heads=8,drop_path=0.2)
        self.post_block=PostBlock(t_dim=768,v_dim=1024)
        self.drop_path0 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.temporal_attn=TemporalAttention(dim=1024, num_heads=8, qkv_bias=False)
        # self.temporal_fc = nn.Linear(1024,1024)
        # self.norm0 = nn.LayerNorm(1024)
        # self.tempFC_1=nn.Conv2d(1024,256,kernel_size=3,padding=1)
        # self.tempFC_2 = nn.Conv2d(256, 64, kernel_size=1)

        self.softmax=nn.Softmax(dim=1)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        for name, m in self.named_modules():
            if 'temporal_conv' in name:
                nn.init.dirac_(m.weight.data) # initialized to be identity
                nn.init.zeros_(m.bias.data)
            if 'temporal_fc' in name:
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _get_local_mask(self, text, split_special_tokens=False):
        output_mask=[]
        doc = nlp(text.lower())
        phrases=[i.text for i in doc._.phrases]
        phrases=sorted(phrases,key=len)
        phrases_dict=build_words_list(phrases)
        split_tokens = []
        if self.tokenizer.do_basic_tokenize:
            basic_tokens=self.tokenizer.basic_tokenizer.tokenize(
                text, never_split=self.tokenizer.all_special_tokens if not split_special_tokens else None
            )
            cnt=0
            for idx,token in enumerate(basic_tokens):
                if token in phrases_dict.keys():
                        if token in self.tokenizer.basic_tokenizer.never_split:
                            split_tokens.append(token)
                            cnt+=1
                        else:
                            out = self.tokenizer.wordpiece_tokenizer.tokenize(token)
                            split_tokens += out
                            cnt += len(out)
                        if isinstance(phrases_dict[token],int):
                            output_mask+=[str(phrases_dict[token])]*cnt
                            cnt=0
                else:
                    cnt=0
                    if token in self.tokenizer.basic_tokenizer.never_split:
                        split_tokens.append(token)
                        output_mask+=["-1"]
                    else:
                        out = self.tokenizer.wordpiece_tokenizer.tokenize(token)
                        split_tokens += out
                        output_mask += ["-1"]*len(out)

                # If the token is part of the never_split set
            if cnt !=0:
                output_mask+=[str(len(phrases)-1)]*cnt

        else:
            split_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(text)
        # res=[]
        # for s,o in zip(split_tokens,output_mask):
        #     res.append(s+"&"+o)
        del split_tokens
        return output_mask
    def get_local_mask(self,caption):
        masks=[]
        for cap in caption:
            masks.append(self._get_local_mask(cap))
        return masks


    def temporal_mean(self,video):
        temporal = []
        for i in range(video.shape[2]):
            image = video[:, :, i, ...]
            image_embeds = self.visual_encoder(image)
            temporal.append(image_embeds.unsqueeze(1))
        temporal = torch.cat(temporal, dim=1)#8,16,197,1024
        return temporal.mean(1)
    def temporal_concate(self,video):
        temporal = []
        for i in range(video.shape[2]):
            image = video[:, :, i, ...]
            image_embeds = self.visual_encoder(image)
            temporal.append(image_embeds.unsqueeze(1))
        temporal = torch.cat(temporal, dim=1)#8,16,197,1024
        return temporal.reshape(temporal.shape[0],-1,temporal.shape[-1])
    def temporal_attention(self,video):
        temporal = []
        for i in range(video.shape[2]):
            image = video[:, :, i, ...]
            image_embeds = self.visual_encoder(image)
            temporal.append(image_embeds.unsqueeze(1))
        temporal = torch.cat(temporal, dim=1)#8,16,197,1024
        temporal_weight=self.softmax(self.tempFC_2(self.tempFC_1(temporal.permute(0,3,1,2))).mean(1)).unsqueeze(-1) #8,16,197,64->8,16,197

        return (temporal*temporal_weight).mean(1)
    def threeDConv(self,video):#3DConv
        temporal=self.visual_encoder(video)
        # temporal = []
        # for i in range(video.shape[2]):
        #     image = video[:, :, i, ...]
        #     image_embeds = self.visual_encoder(image)
        #     temporal.append(image_embeds.unsqueeze(1))
        # temporal = torch.cat(temporal, dim=1)  # 8,16,197,1024
        return self.temporal_block(temporal.reshape(-1,temporal.shape[-2],temporal.shape[-1]),B=temporal.shape[0])

    def temporal_attention_v3(self,video):
        temporal = []
        for i in range(video.shape[2]):
            image = video[:, :, i, ...]
            image_embeds = self.visual_encoder(image)
            temporal.append(image_embeds.unsqueeze(1))
        temporal = torch.cat(temporal, dim=1)#b,t,l,c
        b,t,l,c=temporal.shape
        x = rearrange(temporal, 'b t l c -> (b l) t c')
        x = x + self.temporal_fc(self.drop_path0(self.temporal_attn(self.norm0(x))))
        return x.contiguous().view(b,-1,c)

    def spatial_temporal_image(self,video):
        temporal = []
        for i in range(video.shape[2]):
            image = video[:, :, i, ...]
            image_embeds = self.visual_encoder(image)
            temporal.append(image_embeds.unsqueeze(1))
        x = torch.cat(temporal, dim=1)  # 8,16,197,1024
        # x: B, T, C, H, W
        #B, T, _, L = x.shape
        x = rearrange(x, 'b t h c-> b c t h')
        x = self.proj(x)

        x = rearrange(x, 'b c t h -> b (t h) c')
        x = self.norm(x)

        return x

    # class PatchEmbed(nn.Module):
    #     """ Image to Patch Embedding
    #     """
    #
    #     def __init__(self, patch_size=4, in_chans=3, embed_dim=768):
    #         super().__init__()
    #         patch_size = to_2tuple(patch_size)
    #         self.patch_size = patch_size
    #         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    #         self.norm = nn.LayerNorm(embed_dim)
    #
    #     def forward(self, video):
    #         temporal = []
    #         for i in range(video.shape[2]):
    #             image = video[:, :, i, ...]
    #             image_embeds = self.visual_encoder(image)
    #             temporal.append(image_embeds.unsqueeze(1))
    #         x = torch.cat(temporal, dim=1)  # 8,16,197,1024
    #         # x: B, T, C, H, W
    #         B, T, _, L = x.shape
    #         x = rearrange(x, 'b t c h-> b c t h')
    #         x = self.proj(x)
    #         x = rearrange(x, 'b c t h -> b t c h', b=B, t=T)
    #         x = self.norm(x)
    #         #out_size = ((H * T) // self.patch_size[0], W // self.patch_size[1])
    #
    #         return x#, out_size

    def forward(self, video, caption, mode):

        #assert mode in ['image', 'text', 'multimodal',"image_attn"], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt",padding=True).to(video.device)

        if mode == 'image':
            # return image features
            image_embeds = self.visual_encoder(image)
            return image_embeds

        elif mode == 'text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text')
            return text_output.last_hidden_state
        elif mode=="multimodal_cross":
            image_embeds, frame_embeds = self.threeDConv(video)  # 8,197,1024
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(video.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       return_dict=True, mode='text'
                                       )
            return self.post_block(Et=output.last_hidden_state, Ef=frame_embeds,type="fine_grained")
        elif mode=="multimodal_text":
            image_embeds, frame_embeds = self.threeDConv(video)  # 8,197,1024
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(video.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            return output.last_hidden_state

        elif mode == 'multimodal':
            # return multimodel features

            image_embeds,_ = self.threeDConv(video)#8,197,1024
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(video.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            return self.post_block(Et=output.last_hidden_state,Ev=image_embeds)#8,38,768
        elif mode == 'multimodal_local':
            # return multimodel features
            masks=self.get_local_mask(caption)
            image_embeds,frame_embeds = self.threeDConv(video)  # 8,197,1024
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(video.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            text_feature = output.last_hidden_state
            Et=[]
            for idx,mask in enumerate(masks):
                pre_mask = -1
                feature=[]
                tmp_feature=[]
                for i,m in enumerate(mask):
                    if m=="-1":
                        pre_mask=m
                        if tmp_feature!=[]:
                            feature.append(torch.cat(tmp_feature,dim=0).mean(dim=0).reshape(1,-1))
                            tmp_feature=[]
                        continue
                    else:
                        if m!=pre_mask:
                            if tmp_feature != []:
                                feature.append(torch.cat(tmp_feature, dim=0).mean(dim=0).reshape(1,-1))
                                tmp_feature = []
                        pre_mask=m
                        tmp_feature.append(text_feature[idx,i+1,:].unsqueeze(0))
                if tmp_feature != []:
                    feature.append(torch.cat(tmp_feature, dim=0).mean(dim=0).reshape(1,-1))
                if feature==[]:
                    Et.append(text_feature[idx,0,:].unsqueeze(0))
                else:
                    Et.append(torch.cat(feature,dim=0))


            return self.post_block(Et=(Et,text_feature),Ef=frame_embeds,type="local")
        elif mode == 'multimodal_late':
            # return multimodel features

            image_embeds,frame_embeds = self.threeDConv(video)  # 8,197,1024
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(video.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            return self.post_block(Et=output.last_hidden_state,Ef=frame_embeds,type="fine_grained")
        elif mode == 'multimodal_mix':
            image_embeds, frame_embeds = self.visual_encoder(video)  # 8,197,1024
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(video.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            return self.post_block(Et=output.last_hidden_state,Ev=image_embeds,Ef=frame_embeds, type="mix")
            # 8,38,768
        elif mode=="ThreeDConv_lateFusion":
            temporal=[]
            for i in range(video.shape[2]):
                image = video[:, :, i, ...]
                image_embeds = self.visual_encoder(image)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(video.device)

                text.input_ids[:, 0] = self.tokenizer.enc_token_id
                output = self.text_encoder(text.input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           )
                temporal.append(output.last_hidden_state.unsqueeze(1))
            temporal = torch.cat(temporal, dim=1)  # 8,16,197,1024
            return self.temporal_block(temporal.reshape(-1, temporal.shape[-2], temporal.shape[-1]),
                                       B=temporal.shape[0])

        elif mode=="image_attn":
            image_attn = self.threeDConv(video)
            return image_attn


class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config='BLIP_configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, image, caption):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(
            image.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )
        loss_lm = decoder_output.loss

        return loss_lm

    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions


def blip_decoder(pretrained='', **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model


def blip_feature_extractor(pretrained='', **kwargs):
    model = BLIP_Base(vit="large",**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        #assert (len(msg.missing_keys) == 0)
    return model


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg

class BLIP(nn.Module):
    def __init__(self,type="multimodal"):
        super().__init__()
        self.model = blip_feature_extractor(pretrained="ckpts/model_large.pth")
        self.type=type



    def forward(self, x, text):
        B, C, T, H, W = x.size()
        #x = rearrange(x, 'b c t h w -> (b t) c h w')
        #x = self.model(x, text)
        #x = rearrange(x, '(b t) c h w -> b c t h w', b=B, t=T)
        #return x
        #B,L,C
        return self.model(x,text,self.type).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)

        #return self.model(x, text, "image_attn")

