import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from .blocks import LayerNorm, AffineDropPath
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath, create_act_layer, get_norm_act_layer, create_conv2d

try:
    from .weight_init import trunc_normal_
except:
    from weight_init import trunc_normal_

def type_1_partition(input, partition_size):  # partition_size = [N]
    B, C, T = input.shape
    partitions = input.view(B, C, T // partition_size, partition_size)
    partitions = partitions.permute(0, 2, 3, 1).contiguous().view(-1, partition_size, C)
    return partitions


def type_1_reverse(partitions, original_size, partition_size):  # original_size = [T]
    T = original_size
    B = int(partitions.shape[0] / (T / partition_size))
    output = partitions.view(B, T // partition_size, partition_size, -1)
    output = output.permute(0, 3, 1, 2).contiguous().view(B, -1, T)
    return output



def type_2_partition(input, partition_size):  # partition_size = [M]
    B, C, T = input.shape
    partitions = input.view(B, C, partition_size, T // partition_size)
    partitions = partitions.permute(0, 3, 2, 1).contiguous().view(-1, partition_size, C)
    return partitions


def type_2_reverse(partitions, original_size, partition_size):  # original_size = [T]
    T = original_size
    B = int(partitions.shape[0] / (T / partition_size))
    output = partitions.view(B, T // partition_size, partition_size, -1)
    output = output.permute(0, 3, 2, 1).contiguous().view(B, -1, T)
    return output


''' 1D relative positional bias: B_{h}^{t} '''


def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)

''' MSA '''


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, rel_type, num_heads=16, partition_size=1, attn_drop=0., rel=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.rel_type = rel_type
        self.num_heads = num_heads
        self.partition_size = partition_size
        self.scale = num_heads ** -0.5
        self.attn_area = partition_size
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rel = rel
        if self.rel:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * partition_size - 1), num_heads))
            self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size))
            trunc_normal_(self.relative_position_bias_table, std=.02)
    

    def _get_relative_positional_bias(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.partition_size, self.partition_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, input):
        B_, N, C = input.shape
        qkv = input.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rel:
            attn = attn + self._get_relative_positional_bias()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        return output

''' SkateFormer Block '''


class SkateFormerBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=7, num_heads=16,
                 type_1_size=8, type_2_size=8,
                 attn_drop=0.5, drop=0., rel=True, drop_path=0.2, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SkateFormerBlock, self).__init__()
        self.type_1_size = type_1_size
        self.type_2_size = type_2_size
        self.partition_function = [type_1_partition, type_2_partition]
        self.reverse_function = [type_1_reverse, type_2_reverse]
        self.partition_size = [type_1_size, type_2_size]
        self.rel_type = ['type_1', 'type_2']

        self.norm_1 = norm_layer(in_channels)
        self.mapping = nn.Linear(in_features=in_channels, out_features=2 * in_channels, bias=True)
        # self.gconv = nn.Parameter(torch.zeros(num_heads // (2 * 2), num_points, num_points))
        # trunc_normal_(self.gconv, std=.02)
        self.tconv = nn.Conv1d(in_channels // 2, in_channels // 2, kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2, groups=num_heads // 2)

        # Attention layers
        attention = []
        for i in range(len(self.partition_function)):
            attention.append(
                MultiHeadSelfAttention(in_channels=in_channels // (len(self.partition_function) * 2),
                                       rel_type=self.rel_type[i],
                                       num_heads=num_heads // (len(self.partition_function) * 2),
                                       partition_size=self.partition_size[i], attn_drop=attn_drop, rel=rel))
        self.attention = nn.ModuleList(attention)

        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=int(mlp_ratio * in_channels),
                       act_layer=act_layer, drop=drop)

    def forward(self, input):
        B, C, T = input.shape

        # Partition
        input = input.permute(0, 2, 1).contiguous()
        skip = input

        f = self.mapping(self.norm_1(input)).permute(0, 2, 1).contiguous()

        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2], dim=1)
        y = []

        # G-Conv
        # split_f_conv = torch.chunk(f_conv, 2, dim=1)
        # y_gconv = []
        # split_f_gconv = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)
        # for i in range(self.gconv.shape[0]):
        #     z = torch.einsum('n c t u, v u -> n c t v', split_f_gconv[i], self.gconv[i])
        #     y_gconv.append(z)
        # y.append(torch.cat(y_gconv, dim=1))  # N C T V

        # T-Conv
        y.append(self.tconv(f_conv))

        # Skate-MSA
        split_f_attn = torch.chunk(f_attn, len(self.partition_function), dim=1)

        for i in range(len(self.partition_function)):
            C = split_f_attn[i].shape[1]
            input_partitioned = self.partition_function[i](split_f_attn[i], self.partition_size[i])
            # input_partitioned = input_partitioned.view(-1, self.partition_size[i], C)
            y.append(self.reverse_function[i](self.attention[i](input_partitioned), T, self.partition_size[i]))

        output = self.proj(torch.cat(y, dim=1).permute(0, 2, 1).contiguous())
        output = self.proj_drop(output)
        output = skip + self.drop_path(output)

        # Feed Forward
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        output = output.permute(0, 2, 1).contiguous()
        return output

class DynE_Attn(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            k=1.5,  # k
            init_conv_vars=0.1  # init gaussian variance for the weight
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size
        self.ln = LayerNorm(n_embd)

        self.psi    = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convw  = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)

        
        self.fc        = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
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

    def forward(self, x):

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)

        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))

        convw = self.convw(out)
        convkw = self.convkw(out)

        out = fc * phi + torch.relu(convw + convkw) * psi + out

        return out

class DynESkateformerLayer(nn.Module):
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

        self.gn = nn.GroupNorm(16, n_embd)

        self.dyne_attn = DynE_Attn(n_embd, kernel_size, k, init_conv_vars)

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

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = self.dyne_attn(x)
        
        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()


class DynESkateformerv2Layer(nn.Module):
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

        self.gn = nn.GroupNorm(16, n_embd)

        self.dyne_skate = SkateFormerBlock(n_embd)

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

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()
        out = self.dyne_skate(x)

        return out, out_mask.bool()