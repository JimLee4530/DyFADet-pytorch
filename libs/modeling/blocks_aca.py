import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .weight_init import trunc_normal_
except:
    from weight_init import trunc_normal_

from .blocks import LayerNorm, AffineDropPath    

class ContextGatingBlock(nn.Module):
    def __init__(self, input_dim, kernel_sizes):
        super(ContextGatingBlock, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Conv-Sigmoid 层
        self.conv_sigmoid = nn.Conv1d(2*input_dim * len(kernel_sizes), len(kernel_sizes), kernel_size=1)
        
    def forward(self, x):
        # Extract multiscale features
        multiscale_features = []
        for conv in self.convs:
            feature = F.gelu(conv(x))
            multiscale_features.append(feature)
        
        # Concatenate multiscale features
        multiscale_features_cat = torch.cat(multiscale_features, dim=1)
        
        # Max and Average Pooling
        max_pool = F.adaptive_max_pool1d(multiscale_features_cat, output_size=1)
        avg_pool = F.adaptive_avg_pool1d(multiscale_features_cat, output_size=1)
        
        
        # Concatenate max and avg pooling
        pooled_features = torch.cat([max_pool, avg_pool], dim=1)
        
        # Gating coefficients
        gating_weights = torch.sigmoid(self.conv_sigmoid(pooled_features))
        
        # Apply gating
        gated_features = []
        for idx, multiscale_feature in enumerate(multiscale_features):
            gated_features.append(multiscale_feature * gating_weights[:,idx,:].unsqueeze(-1))

        
        return torch.sum(torch.stack(gated_features), dim=0)

class ContextAttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(ContextAttentionModule, self).__init__()
        
        # K-branch
        self.k_branch = nn.Linear(input_dim, input_dim)
        
        # Q-branch
        self.q_branch = nn.Linear(input_dim, input_dim)
        
        # Context Gating Block
        self.cgb = ContextGatingBlock(input_dim, kernel_sizes=[1, 3, 5])
        
    def forward(self, x):
        # K-branch
        k_features = self.k_branch(x.transpose(1, 2)).transpose(1, 2)
        
        # Q-branch
        q_features = self.q_branch(x.transpose(1, 2)).transpose(1, 2)
        
        # Context Gating Block
        gated_features = self.cgb(q_features)
        
        # Modulate K-features with gated attention
        cam_output = gated_features * k_features
        
        return cam_output

class ContextAttentionModulev2(nn.Module):
    def __init__(self, input_dim, init_conv_vars):
        super(ContextAttentionModulev2, self).__init__()
        
        # K-branch
        self.k_branch = nn.Conv1d(input_dim, input_dim, 1, stride=1, padding=0, groups=input_dim)
        
        # Q-branch
        self.q_branch = nn.Conv1d(input_dim, input_dim, 1, stride=1, padding=0, groups=input_dim)
        
        # Context Gating Block
        self.cgb = ContextGatingBlock(input_dim, kernel_sizes=[1, 3, 5])
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.k_branch.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.q_branch.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.k_branch.bias, 0)
        torch.nn.init.constant_(self.q_branch.bias, 0)
        
    def forward(self, x):
        # K-branch
        k_features = self.k_branch(x)
        
        # Q-branch
        q_features = self.q_branch(x)
        
        # Context Gating Block
        gated_features = self.cgb(q_features)
        
        # Modulate K-features with gated attention
        cam_output = gated_features * k_features
        
        return cam_output

class DynELayer_aca(nn.Module):
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

        # Context Attention Module (CAM)
        self.cam = ContextAttentionModulev2(n_embd, init_conv_vars)

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

        cam_output = self.cam(out)

        out = fc * phi + torch.relu(convw + convkw) * psi + out + cam_output
        
        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()
    

if __name__ == "__main__":
    import torch 