import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from efficientnet_pytorch.model import MemoryEfficientSwish

from .blocks import LayerNorm, AffineDropPath

class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv1d(dim, dim, 1, 1, 0),
                            MemoryEfficientSwish(),
                            nn.Conv1d(dim, dim, 1, 1, 0)
                            #nn.Identity()
                         )
    def forward(self, x):
        return self.act_block(x)

class DynELayerv5(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_sizes=[7, 5],  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used in mlp,
            init_conv_vars=0.1  # init gaussian variance for the weight
    ):
        super().__init__()

        self.stride = n_ds_stride
        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        # add 1 to avoid have the same size as the instant-level branch

        self.fc        = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # local
        self.kernel_sizes = kernel_sizes
        self.dim = n_embd
        self.num_heads = len(kernel_sizes)
        self.dim_head = n_embd // self.num_heads
        self.scalor = self.dim_head ** -0.5
        self.convs = nn.ModuleList()
        self.act_blocks = nn.ModuleList()
        self.qkvs = nn.ModuleList()
        qkv_bias = True 
        attn_drop = 0
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            self.convs.append(nn.Conv1d(3*self.dim_head, 3*self.dim_head, kernel_size,
                         1, kernel_size//2, groups=3*self.dim_head))
            self.act_blocks.append(AttnMap(self.dim_head))
            self.qkvs.append(nn.Conv1d(self.dim, 3*self.dim_head, 1, 1, 0, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)

        self.merge_fc = nn.Conv1d(2*n_embd, n_embd, 1, stride=1, padding=0)

        if n_ds_stride > 1:
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c l)
        '''
        b, c, l = x.size()
        qkv = to_qkv(x) #(b (3 m d) l)
        qkv = mixer(qkv).reshape(b, 3, -1, l).transpose(0, 1).contiguous() #(3 b (m d) l)
        q, k, v = qkv #(b (m d) l)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res =attn.mul(v) #(b (m d) l)
        return res

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = self.ln(x)

        fc = self.fc(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))

        res = []
        for i in range(len(self.kernel_sizes)):
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        res.append(fc*phi)
        out = self.merge_fc(torch.cat(res, dim=1)) + out
        
        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool() 

class SGPBlock_cloformer(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_sizes=[5, 7],  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            downsample_type='max',
            init_conv_vars=1  # init gaussian variance for the weight
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        # self.kernel_size = kernel_size
        self.stride = n_ds_stride

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        # assert kernel_size % 2 == 1
        # # add 1 to avoid have the same size as the instant-level branch
        # up_size = round((kernel_size + 1) * k)
        # up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # local
        self.kernel_sizes = kernel_sizes
        self.dim = n_embd
        self.num_heads = len(kernel_sizes)
        self.dim_head = n_embd // self.num_heads
        self.scalor = self.dim_head ** -0.5
        self.convs = nn.ModuleList()
        self.act_blocks = nn.ModuleList()
        self.qkvs = nn.ModuleList()
        qkv_bias = True 
        attn_drop = 0
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            self.convs.append(nn.Conv1d(3*self.dim_head, 3*self.dim_head, kernel_size,
                         1, kernel_size//2, groups=3*self.dim_head))
            self.act_blocks.append(AttnMap(self.dim_head))
            self.qkvs.append(nn.Conv1d(self.dim, 3*self.dim_head, 1, 1, 0, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)

        self.merge_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0)

        # input
        if n_ds_stride > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

        torch.nn.init.normal_(self.merge_fc.weight, 0, init_conv_vars)
        for conv in self.convs:
            torch.nn.init.normal_(conv.weight, 0, init_conv_vars)
            torch.nn.init.constant_(conv.bias, 0)
        for conv in self.qkvs:
            torch.nn.init.normal_(conv.weight, 0, init_conv_vars)
            torch.nn.init.constant_(conv.bias, 0)
        for act_block in self.act_blocks:
            torch.nn.init.normal_(act_block.act_block[0].weight, 0, init_conv_vars)
            torch.nn.init.constant_(act_block.act_block[0].bias, 0)
            torch.nn.init.normal_(act_block.act_block[2].weight, 0, init_conv_vars)
            torch.nn.init.constant_(act_block.act_block[2].bias, 0)
        torch.nn.init.constant_(self.merge_fc.bias, 0)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c l)
        '''
        b, c, l = x.size()
        qkv = to_qkv(x) #(b (3 m d) l)
        qkv = mixer(qkv).reshape(b, 3, -1, l).transpose(0, 1).contiguous() #(3 b (m d) l)
        q, k, v = qkv #(b (m d) l)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res =attn.mul(v) #(b (m d) l)
        return res

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = self.ln(x)
        fc = self.fc(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))

        res = []
        for i in range(len(self.kernel_sizes)):
            res.append(self.high_fre_attntion(out, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        # res.append(fc*phi)
        out = self.merge_fc(torch.cat(res, dim=1))+fc*phi + out

        # out = fc * phi + (convw + convkw) * psi + out

        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()