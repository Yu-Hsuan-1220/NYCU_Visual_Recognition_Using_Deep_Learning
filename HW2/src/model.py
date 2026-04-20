"""Deformable DETR: ResNet-50 multi-scale backbone, deformable attention, prediction heads.

Pure PyTorch implementation (no custom CUDA ops required).
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Frozen BatchNorm (used in backbone during fine-tuning)
# ---------------------------------------------------------------------------

class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed running stats and affine parameters."""

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + 1e-5).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def _replace_bn_with_frozen(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features)
            frozen.weight.copy_(child.weight)
            frozen.bias.copy_(child.bias)
            frozen.running_mean.copy_(child.running_mean)
            frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
        else:
            _replace_bn_with_frozen(child)


# ---------------------------------------------------------------------------
# Multi-Scale Backbone
# ---------------------------------------------------------------------------

class MultiScaleBackbone(nn.Module):
    """ResNet-50 backbone returning multi-scale features (strides 8, 16, 32)."""

    def __init__(self, pretrained=True, frozen_bn=True, freeze_at=1):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None,
        )
        if frozen_bn:
            _replace_bn_with_frozen(resnet)

        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1   # stride 4,  256 ch
        self.layer2 = resnet.layer2   # stride 8,  512 ch
        self.layer3 = resnet.layer3   # stride 16, 1024 ch
        self.layer4 = resnet.layer4   # stride 32, 2048 ch
        self.num_channels = [512, 1024, 2048]

        if freeze_at >= 1:
            for p in self.layer0.parameters():
                p.requires_grad_(False)
            for p in self.layer1.parameters():
                p.requires_grad_(False)
        if freeze_at >= 2:
            for p in self.layer2.parameters():
                p.requires_grad_(False)

    def forward(self, x, mask):
        x = self.layer0(x)
        c2 = self.layer1(x)    # stride 4,  256 ch
        c3 = self.layer2(c2)   # stride 8,  512 ch
        c4 = self.layer3(c3)   # stride 16, 1024 ch
        c5 = self.layer4(c4)   # stride 32, 2048 ch

        features = [c3, c4, c5]
        masks = []
        for feat in features:
            m = F.interpolate(
                mask.unsqueeze(1).float(), size=feat.shape[-2:]
            ).bool().squeeze(1)
            masks.append(m)
        return features, masks


# ---------------------------------------------------------------------------
# Positional encoding (2-D sine)
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


# ---------------------------------------------------------------------------
# Multi-Scale Deformable Attention (pure PyTorch)
# ---------------------------------------------------------------------------

def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                sampling_locations, attention_weights):
    """Pure PyTorch multi-scale deformable attention core.

    Args:
        value: (B, sum(H_l*W_l), n_heads, head_dim)
        value_spatial_shapes: (n_levels, 2) tensor [H, W]
        sampling_locations: (B, Len_q, n_heads, n_levels, n_points, 2) in [0,1]
        attention_weights: (B, Len_q, n_heads, n_levels, n_points)
    Returns:
        output: (B, Len_q, n_heads * head_dim)
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, _, L_, P_, _ = sampling_locations.shape

    value_list = value.split(
        [H_.item() * W_.item() for H_, W_ in value_spatial_shapes], dim=1,
    )
    sampling_grids = 2 * sampling_locations - 1  # [0,1] -> [-1,1]

    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = H_.item(), W_.item()
        # (N_, H_*W_, M_, D_) -> (N_*M_, D_, H_, W_)
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2)
            .reshape(N_ * M_, D_, H_, W_)
        )
        # (N_, Lq_, M_, P_, 2) -> (N_*M_, Lq_, P_, 2)
        sampling_grid_l_ = (
            sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        )
        # (N_*M_, D_, Lq_, P_)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # (N_, Lq_, M_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = (
        attention_weights.transpose(1, 2)
        .reshape(N_ * M_, 1, Lq_, L_ * P_)
    )
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2)
        * attention_weights
    ).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2,
        )
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points,
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32
        ) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        Args:
            query: (B, Len_q, d_model)
            reference_points: (B, Len_q, n_levels, 2) normalized [0,1]
            input_flatten: (B, sum(H_l*W_l), d_model)
            input_spatial_shapes: (n_levels, 2) tensor [H, W]
            input_level_start_index: (n_levels,)
            input_padding_mask: (B, sum(H_l*W_l)) True = padding
        """
        B, Len_q, _ = query.shape
        B, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(B, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        # Normalize offsets by spatial shape
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )  # (n_levels, 2) [W, H]
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets
            / offset_normalizer[None, None, None, :, None, :]
        )

        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights,
        )
        return self.output_proj(output)


# ---------------------------------------------------------------------------
# Deformable Transformer Encoder
# ---------------------------------------------------------------------------

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes,
                level_start_index, padding_mask=None):
        # Deformable self-attention
        src2 = self.self_attn(
            src + pos, reference_points, src,
            spatial_shapes, level_start_index, padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Generate reference points for each spatial position at each level."""
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij',
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # (B, H*W, 2)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # (B, sum(H*W), 2)
        # Expand to all levels
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points  # (B, sum(H*W), n_levels, 2)

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios,
                pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device,
        )
        for layer in self.layers:
            output = layer(
                output, pos, reference_points, spatial_shapes,
                level_start_index, padding_mask,
            )
        return output


# ---------------------------------------------------------------------------
# Deformable Transformer Decoder
# ---------------------------------------------------------------------------

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # Self-attention (standard multi-head, not deformable)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # Cross-attention (deformable)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_points, src,
                src_spatial_shapes, level_start_index,
                src_padding_mask=None):
        # Self-attention among queries
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Cross-attention with multi-scale encoder features
        tgt2 = self.cross_attn(
            tgt + query_pos, reference_points, src,
            src_spatial_shapes, level_start_index, src_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None  # set externally for iterative box refinement

    def forward(self, tgt, reference_points, src, src_spatial_shapes,
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # Scale reference points by valid ratios for each level
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )

            output = layer(
                output, query_pos, reference_points_input, src,
                src_spatial_shapes, src_level_start_index, src_padding_mask,
            )

            # Iterative box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    new_reference_points = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output.unsqueeze(0), reference_points.unsqueeze(0)


# ---------------------------------------------------------------------------
# Deformable Transformer
# ---------------------------------------------------------------------------

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 num_feature_levels=4, enc_n_points=4, dec_n_points=4,
                 return_intermediate_dec=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        enc_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout,
            num_feature_levels, nhead, enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout,
            num_feature_levels, nhead, dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            dec_layer, num_decoder_layers,
            return_intermediate=return_intermediate_dec,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points_head = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.xavier_uniform_(self.reference_points_head.weight.data, gain=1.0)
        nn.init.constant_(self.reference_points_head.bias.data, 0.0)
        nn.init.normal_(self.level_embed)

    @staticmethod
    def get_valid_ratio(mask):
        """Fraction of valid (non-padding) area for a feature map."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)  # (B, 2)

    def forward(self, srcs, masks, pos_embeds, query_embed):
        """
        Args:
            srcs: list of (B, C, H_l, W_l) projected multi-scale features
            masks: list of (B, H_l, W_l) padding masks
            pos_embeds: list of (B, C, H_l, W_l) positional encodings
            query_embed: (num_queries, hidden_dim*2) query + positional
        Returns:
            hs: (num_dec_layers, B, Q, C)
            init_reference: (B, Q, 2)
            inter_references: (num_dec_layers, B, Q, 2 or 4)
        """
        # Flatten multi-scale features
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)          # (B, H*W, C)
            mask = mask.flatten(1)                         # (B, H*W)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # (B, H*W, C)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)            # (B, sum(H*W), C)
        mask_flatten = torch.cat(mask_flatten, 1)          # (B, sum(H*W))
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device,
        )
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], 1,
        )  # (B, n_levels, 2)

        # Encoder
        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index, valid_ratios,
            lvl_pos_embed_flatten, mask_flatten,
        )

        # Prepare decoder input
        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points_head(query_embed).sigmoid()
        init_reference_out = reference_points

        # Decoder
        hs, inter_references = self.decoder(
            tgt, reference_points, memory, spatial_shapes,
            level_start_index, valid_ratios, query_embed, mask_flatten,
        )

        return hs, init_reference_out, inter_references


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
        )
        self.num_layers = num_layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# Deformable DETR
# ---------------------------------------------------------------------------

class DeformableDETR(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, num_queries=50,
                 num_feature_levels=4, enc_n_points=4, dec_n_points=4,
                 aux_loss=True, with_box_refine=False,
                 pretrained_backbone=True, freeze_at=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        # Backbone
        self.backbone = MultiScaleBackbone(
            pretrained=pretrained_backbone, freeze_at=freeze_at,
        )

        # Input projections for backbone features
        backbone_channels = self.backbone.num_channels  # [256, 512, 1024, 2048]
        input_proj_list = []
        for ch in backbone_channels:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(ch, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        # Additional feature levels via stride-2 convolutions
        in_channels = backbone_channels[-1]
        for _ in range(num_feature_levels - len(backbone_channels)):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3,
                          stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)

        # Positional encoding
        self.position_encoding = PositionEmbeddingSine(hidden_dim // 2)

        # Transformer
        self.transformer = DeformableTransformer(
            d_model=hidden_dim, nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_feature_levels=num_feature_levels,
            enc_n_points=enc_n_points,
            dec_n_points=dec_n_points,
            return_intermediate_dec=True,
        )

        # Query embedding (split into content + positional parts)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        # Prediction heads
        _class_embed = nn.Linear(hidden_dim, num_classes + 1)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        # Initialize bbox last layer for stable training
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if with_box_refine:
            self.class_embed = _get_clones(_class_embed, num_decoder_layers)
            self.bbox_embed = _get_clones(_bbox_embed, num_decoder_layers)
            # Bias w,h towards small boxes for the first layer
            nn.init.constant_(
                self.bbox_embed[0].layers[-1].bias.data[2:], -2.0,
            )
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(_bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [_class_embed for _ in range(num_decoder_layers)]
            )
            self.bbox_embed = nn.ModuleList(
                [_bbox_embed for _ in range(num_decoder_layers)]
            )
            self.transformer.decoder.bbox_embed = None

        # Init input projections
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, images, masks):
        # 1. Backbone
        features, feature_masks = self.backbone(images, masks)

        # 2. Project features + create additional levels
        srcs = []
        proj_masks = []
        pos_embeds = []
        for l in range(len(features)):
            src = self.input_proj[l](features[l])
            srcs.append(src)
            proj_masks.append(feature_masks[l])
            pos_embeds.append(self.position_encoding(feature_masks[l]))

        if self.num_feature_levels > len(features):
            for l in range(len(features), self.num_feature_levels):
                if l == len(features):
                    src = self.input_proj[l](features[-1])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = F.interpolate(
                    masks.unsqueeze(1).float(), size=src.shape[-2:]
                ).bool().squeeze(1)
                srcs.append(src)
                proj_masks.append(m)
                pos_embeds.append(self.position_encoding(m))

        # 3. Transformer
        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references = self.transformer(
            srcs, proj_masks, pos_embeds, query_embeds,
        )
        # hs: (num_dec_layers, B, Q, hidden_dim)
        # init_reference: (B, Q, 2)
        # inter_references: (num_dec_layers, B, Q, 2)

        # 4. Predictions (residual bbox prediction relative to reference points)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": c, "pred_boxes": b}
                for c, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out


def build_model(args):
    return DeformableDETR(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        pretrained_backbone=args.pretrained_backbone,
        freeze_at=args.freeze_at,
    )
