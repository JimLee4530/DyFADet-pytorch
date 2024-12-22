import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, Scale, LayerNorm, DTFAM

class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, y, z, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(y, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(z, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class DynamicScale_transformer(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        num_convs: int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        num_adjacent_scales: int = 2,
        depth_module: nn.Module = None,
        norm: str = "GN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None
    ):
        super(DynamicScale_transformer, self).__init__()
        self.num_groups = num_groups
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module
        dynamic_convs = [
            DTFAM(dim=in_channels, o_dim= in_channels, ka=kernel_size, gate_activation="GeReTanH",
                gate_activation_kargs = gate_activation_kargs)
            
 
        for _ in range(num_adjacent_scales)]
        
        self.dynamic_convs = nn.ModuleList(dynamic_convs)
        
        self.resize = lambda x, s : F.interpolate(
            x, size=s, mode="nearest")
        
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        self.init_parameters()

        self.norm = LayerNorm(out_channels)
        self.act = nn.ReLU()

        self.transformer = MaskedMHCA(
            in_channels,
            4,
            n_qx_stride=1,
            n_kv_stride=1,
            attn_pdrop=0.0,
            proj_pdrop=0.0
        )

    def init_parameters(self):
        for module in self.dynamic_convs:
            module.init_parameters()

    def forward(self, inputs, fpn_masks):

        dynamic_scales = []
        for l, x in enumerate(inputs):
            dynamic_scales.append([m(x, fpn_masks[l])[0] for m in self.dynamic_convs])
        
        outputs = []
        out_masks = []
        for l, x in enumerate(inputs):
            scale_feature = []
            
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                if l_source >= 0 and l_source < len(inputs):
                    
                    feature = self.resize(dynamic_scales[l_source][s], x.shape[-1:])
                    scale_feature.append(feature)
            
            if len(scale_feature) == 1:
                scale_feature = self.transformer(scale_feature[0], x, x, fpn_masks[l])[0] * self.scale_weight + x * self.output_weight
            else:
                scale_feature = self.transformer(scale_feature[0], scale_feature[1], x, fpn_masks[l])[0]* self.scale_weight + x * self.output_weight
            # scale_feature = sum(scale_feature) * self.scale_weight + x * self.output_weight
            
            if self.depth_module is not None:
                scale_feature, masks = self.depth_module(scale_feature, fpn_masks[l])

            outputs.append(scale_feature)
                 
        out_masks = fpn_masks
        
        return outputs, out_masks   
    
class DynamicScale_transformerv2(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        num_convs: int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        num_adjacent_scales: int = 2,
        depth_module: nn.Module = None,
        norm: str = "GN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None
    ):
        super(DynamicScale_transformerv2, self).__init__()
        self.num_groups = num_groups
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module
        dynamic_convs = [
            DTFAM(dim=in_channels, o_dim= in_channels, ka=kernel_size, gate_activation="GeReTanH",
                gate_activation_kargs = gate_activation_kargs)
            
 
        for _ in range(num_adjacent_scales)]
        
        self.dynamic_convs = nn.ModuleList(dynamic_convs)
        
        self.resize = lambda x, s : F.interpolate(
            x, size=s, mode="nearest")
        
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        self.init_parameters()

        self.norm = LayerNorm(out_channels)
        self.act = nn.ReLU()

        self.transformer = MaskedMHCA(
            in_channels,
            4,
            n_qx_stride=1,
            n_kv_stride=1,
            attn_pdrop=0.0,
            proj_pdrop=0.0
        )

    def init_parameters(self):
        for module in self.dynamic_convs:
            module.init_parameters()

    def forward(self, inputs, fpn_masks):

        dynamic_scales = []
        for l, x in enumerate(inputs):
            dynamic_scales.append([m(x, fpn_masks[l])[0] for m in self.dynamic_convs])
        
        outputs = []
        out_masks = []
        for l, x in enumerate(inputs):
            scale_feature = []
            
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                if l_source >= 0 and l_source < len(inputs):
                    
                    feature = self.resize(dynamic_scales[l_source][s], x.shape[-1:])
                    scale_feature.append(feature)
            
            if len(scale_feature) == 1:
                scale_feature = self.transformer(scale_feature[0], x, x, fpn_masks[l])[0] * self.scale_weight + x * self.output_weight
            else:
                scale_feature = self.transformer(scale_feature[0], x, x, fpn_masks[l])[0]* self.scale_weight + x * self.output_weight
            # scale_feature = sum(scale_feature) * self.scale_weight + x * self.output_weight
            
            if self.depth_module is not None:
                scale_feature, masks = self.depth_module(scale_feature, fpn_masks[l])

            outputs.append(scale_feature)
                 
        out_masks = fpn_masks
        
        return outputs, out_masks   
    
class DynamicScale_transformerv3(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        num_convs: int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        num_adjacent_scales: int = 2,
        depth_module: nn.Module = None,
        norm: str = "GN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None
    ):
        super(DynamicScale_transformerv3, self).__init__()
        self.num_groups = num_groups
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module
        dynamic_convs = [
            DTFAM(dim=in_channels, o_dim= in_channels, ka=kernel_size, gate_activation="GeReTanH",
                gate_activation_kargs = gate_activation_kargs)
            
 
        for _ in range(num_adjacent_scales)]
        
        self.dynamic_convs = nn.ModuleList(dynamic_convs)
        
        self.resize = lambda x, s : F.interpolate(
            x, size=s, mode="nearest")
        
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        self.init_parameters()

        self.norm = LayerNorm(out_channels)
        self.act = nn.ReLU()

        self.transformer = MaskedMHCA(
            in_channels,
            4,
            n_qx_stride=1,
            n_kv_stride=1,
            attn_pdrop=0.0,
            proj_pdrop=0.0
        )

    def init_parameters(self):
        for module in self.dynamic_convs:
            module.init_parameters()

    def forward(self, inputs, fpn_masks):

        dynamic_scales = []
        for l, x in enumerate(inputs):
            dynamic_scales.append([m(x, fpn_masks[l])[0] for m in self.dynamic_convs])
        
        outputs = []
        out_masks = []
        for l, x in enumerate(inputs):
            scale_feature = []
            
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                if l_source >= 0 and l_source < len(inputs):
                    
                    feature = self.resize(dynamic_scales[l_source][s], x.shape[-1:])
                    scale_feature.append(feature)
            
            if len(scale_feature) == 1:
                scale_feature = self.transformer(x, scale_feature[0], x, fpn_masks[l])[0] * self.scale_weight + x * self.output_weight
            else:
                scale_feature = self.transformer(scale_feature[1], scale_feature[0], x, fpn_masks[l])[0]* self.scale_weight + x * self.output_weight
            # scale_feature = sum(scale_feature) * self.scale_weight + x * self.output_weight
            
            if self.depth_module is not None:
                scale_feature, masks = self.depth_module(scale_feature, fpn_masks[l])

            outputs.append(scale_feature)
                 
        out_masks = fpn_masks
        
        return outputs, out_masks   

# dyn_transformer_heads.py
class TDynPtTransformerClsHeadv2(nn.Module):
    """
    Shared 1D MSDy-head for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=5,
        act_layer=nn.ReLU,
        empty_cls = [],
        gate_activation_kargs: dict = None
    ):
        super().__init__()
        self.act = act_layer()
        
        assert num_layers-1 >0

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()

        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            cls_subnet_conv = DTFAM(dim=in_dim, o_dim= feat_dim, ka=kernel_size, conv_type = 'others', gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs)

            self.head.append(
                DynamicScale_transformer(
                in_dim,
                out_dim,
                num_convs=1,
                kernel_size=kernel_size,
                padding=1,
                stride=kernel_size // 2,
                num_groups=1,
                num_adjacent_scales=2,
                depth_module=cls_subnet_conv,
                gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs))

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        feats = fpn_feats
        for i in range(len(self.head)):
            feats, fpn_masks = self.head[i](feats, fpn_masks)
            
            for j in range(len(feats)):
                feats[j] =  self.act(feats[j])

        for cur_out, cur_mask in zip(feats, fpn_masks):
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        return out_logits
    
class TDynPtTransformerRegHeadv2(nn.Module):
    """
    Shared 1D MSDy-head for regression
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=5,
        act_layer=nn.ReLU,
        gate_activation_kargs: dict = None
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()
        
        assert num_layers-1 > 0

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            reg_subnet_conv = DTFAM(dim=in_dim, o_dim= feat_dim, ka=kernel_size, conv_type = 'others', gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs)

            self.head.append(DynamicScale_transformer(
                in_dim,
                out_dim,
                num_convs=1,
                kernel_size=kernel_size,
                padding=1,
                stride=kernel_size // 2,
                num_groups=1,
                num_adjacent_scales=2,
                depth_module=reg_subnet_conv,
                gate_activation=gate_activation_kargs['type'],
                gate_activation_kargs = gate_activation_kargs)
                )

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )


    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels
        
        # apply the classifier for each pyramid level
        out_offsets = tuple()
        feats = fpn_feats
        for i in range(len(self.head)):
            
            feats, fpn_masks = self.head[i](feats, fpn_masks)
            for j in range(len(feats)):
                feats[j] = self.act(feats[j])
        
        for l in range(self.fpn_levels):
            cur_offsets, _  = self.offset_head(feats[l], fpn_masks[l])            
            out_offsets +=  ( F.relu(self.scale[l](cur_offsets)) , )
               
        return out_offsets 