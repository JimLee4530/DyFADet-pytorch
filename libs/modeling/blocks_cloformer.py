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

class DynELayerv2(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
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

        self.kernel_size = kernel_size
        self.stride = n_ds_stride
        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi    = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convw  = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)

        
        self.fc        = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        self.merge_fc = nn.Conv1d(2*n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

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

        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

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
        psi = self.psi(out)
        fc = self.fc(out)

        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))

        convw = self.convw(out)
        convkw = self.convkw(out)

        out = self.merge_fc(torch.cat([fc * phi , torch.relu(convw + convkw) * psi], dim=1)) + out
        
        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool() 
    
class DynELayerv3(nn.Module):
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
        self.num_heads = 4
        self.dim_head = n_embd // 4
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

        self.merge_fc = nn.Conv1d(len(kernel_sizes)*self.dim_head+n_embd, n_embd, 1, stride=1, padding=0)

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
        res = attn.mul(v) #(b (m d) l)
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
    
class DynELayerv4(nn.Module):
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
        # self.num_heads = len(kernel_sizes)
        # self.dim_head = n_embd // self.num_heads
        self.scalor = self.dim ** -0.5
        self.convs = nn.ModuleList()
        self.act_blocks = nn.ModuleList()
        self.qkvs = nn.ModuleList()
        qkv_bias = True 
        attn_drop = 0
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            self.convs.append(nn.Conv1d(3*self.dim, 3*self.dim, kernel_size,
                         1, kernel_size//2, groups=3*self.dim))
            self.act_blocks.append(AttnMap(self.dim))
            self.qkvs.append(nn.Conv1d(self.dim, 3*self.dim, 1, 1, 0, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)

        self.merge_fc = nn.Conv1d((len(kernel_sizes)+1)*n_embd, n_embd, 1, stride=1, padding=0, bias=qkv_bias)

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
        res = attn.mul(v) #(b (m d) l)
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
        res = attn.mul(v) #(b (m d) l)
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

class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, stride,
                 out_channels, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Conv1d(in_channels, hidden_channels, 1, 1, 0)
        self.act = act_layer()
        self.dwconv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, stride, 
                                kernel_size//2, groups=hidden_channels)
        self.fc2 = nn.Conv1d(hidden_channels, out_channels, 1, 1, 0)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x) # 需要 验证
        return x

# convFFN + maxpool downsampler
class DynELayerv6(nn.Module):
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

        # self.mlp = nn.Sequential(
        #     nn.Conv1d(n_embd, n_hidden, 1, groups=group),
        #     act_layer(),
        #     nn.Conv1d(n_hidden, n_out, 1, groups=group),
        # )
        # self.mlp = ConvFFN(dim, mlp_hidden_dim, mlp_kernel_size, stride, out_dim, 
        #                 drop_out=mlp_drop)
        self.mlp = ConvFFN(n_embd, n_hidden, 1, 1, n_out, act_layer=act_layer, drop_out=path_pdrop)

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
        res = attn.mul(v) #(b (m d) l)
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

# convFFN + dwconv downsampler
class DynELayerv7(nn.Module):
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
                # self.downsample = nn.Sequential(
                #                 nn.Conv1d(self.dim, self.dim, kernel_size, stride, padding, groups=self.dim),
                #                 nn.BatchNorm1d(self.dim),
                #                 nn.Conv1d(self.dim, self.dim, 1, 1, 0, groups=self.dim),
                #             )
                self.stride = stride
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        # self.mlp = nn.Sequential(
        #     nn.Conv1d(n_embd, n_hidden, 1, groups=group),
        #     act_layer(),
        #     nn.Conv1d(n_hidden, n_out, 1, groups=group),
        # )
        # self.mlp = ConvFFN(dim, mlp_hidden_dim, mlp_kernel_size, stride, out_dim, 
        #                 drop_out=mlp_drop)
        self.mlp = ConvFFN(n_embd, n_hidden, kernel_size, self.stride, n_out, act_layer=act_layer, drop_out=path_pdrop)

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
        res = attn.mul(v) #(b (m d) l)
        return res

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape

        out = self.ln(x)

        fc = self.fc(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))

        res = []
        for i in range(len(self.kernel_sizes)):
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        res.append(fc*phi)
        out = self.merge_fc(torch.cat(res, dim=1)) + out
        
        out = x*mask + self.drop_path_out(out)
        out_down = self.downsample(out)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()
        # FFN
        out = out_down + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool() 

if __name__ == "__main__":
    import torch 