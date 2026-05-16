"""PromptIR architecture with NAFNet blocks as the internal encoder/decoder unit.

Original PromptIR: Potlapalli et al., "PromptIR: Prompting for All-in-One
Image Restoration", NeurIPS 2023 (arXiv:2306.13090).

This file keeps PromptIR's four-level U-Net topology and the three
PromptGenBlock injection sites, but replaces the internal TransformerBlock
(MDTA + GDFN) with NAFBlock from NAFNet (Chen et al., ECCV 2022).

Channel widths are derived from ``dim`` and ``prompt_dims`` instead of the
original hardcoded constants, so ``dim`` is freely tunable as long as every
resulting width is even (SimpleGate halves channels).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafblock import NAFBlock


class _Block(nn.Module):
    """Thin wrapper so we could swap NAFBlock for something else later."""

    def __init__(self, dim, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        self.block = NAFBlock(dim, dw_expand=dw_expand, ffn_expand=ffn_expand,
                              drop_out_rate=drop_out_rate)

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class PromptGenBlock(nn.Module):
    """Verbatim from PromptIR (Potlapalli et al., NeurIPS 2023)."""

    def __init__(self, prompt_dim, prompt_len, prompt_size, lin_dim):
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
            self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt


class PromptIRNAF(nn.Module):
    """PromptIR with NAFBlock as the internal encoder/decoder block."""

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        dw_expand=2,
        ffn_expand=2,
        drop_path=0.0,
        decoder=True,
        prompt_dims=(64, 128, 320),
        prompt_len=5,
        prompt_sizes=(64, 32, 16),
        bias=False,
    ):
        super().__init__()
        self.decoder = decoder

        c1 = dim
        c2 = dim * 2
        c3 = dim * 4
        c4 = dim * 8

        pd1, pd2, pd3 = prompt_dims
        ps1, ps2, ps3 = prompt_sizes

        def blk(d):
            return _Block(d, dw_expand=dw_expand, ffn_expand=ffn_expand,
                          drop_out_rate=drop_path)

        self.patch_embed = OverlapPatchEmbed(inp_channels, c1)

        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=pd1, prompt_len=prompt_len,
                                          prompt_size=ps1, lin_dim=c2)
            self.prompt2 = PromptGenBlock(prompt_dim=pd2, prompt_len=prompt_len,
                                          prompt_size=ps2, lin_dim=c3)
            self.prompt3 = PromptGenBlock(prompt_dim=pd3, prompt_len=prompt_len,
                                          prompt_size=ps3, lin_dim=c4)

        # encoder
        self.encoder_level1 = nn.Sequential(*[blk(c1) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(c1)

        self.encoder_level2 = nn.Sequential(*[blk(c2) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(c2)

        self.encoder_level3 = nn.Sequential(*[blk(c3) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(c3)

        self.latent = nn.Sequential(*[blk(c4) for _ in range(num_blocks[3])])

        # decoder level3: prompt3 fuses INTO latent
        if self.decoder:
            self.noise_level3 = blk(c4 + pd3)
            self.reduce_noise_level3 = nn.Conv2d(c4 + pd3, c3, kernel_size=1, bias=bias)
        else:
            # ablation: bring latent down to c3 so up4_3 still fits
            self.latent_reduce = nn.Conv2d(c4, c3, kernel_size=1, bias=bias)
        self.up4_3 = Upsample(c3)
        self.reduce_chan_level3 = nn.Conv2d(c2 + c3, c3, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[blk(c3) for _ in range(num_blocks[2])])

        # decoder level2: prompt2 fuses INTO decoder_level3 output
        if self.decoder:
            self.noise_level2 = blk(c3 + pd2)
            self.reduce_noise_level2 = nn.Conv2d(c3 + pd2, c3, kernel_size=1, bias=bias)
        self.up3_2 = Upsample(c3)
        self.reduce_chan_level2 = nn.Conv2d(c2 + c2, c2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[blk(c2) for _ in range(num_blocks[1])])

        # decoder level1: prompt1 fuses INTO decoder_level2 output
        if self.decoder:
            self.noise_level1 = blk(c2 + pd1)
            self.reduce_noise_level1 = nn.Conv2d(c2 + pd1, c2, kernel_size=1, bias=bias)
        self.up2_1 = Upsample(c2)
        self.decoder_level1 = nn.Sequential(*[blk(c2) for _ in range(num_blocks[0])])

        self.refinement = nn.Sequential(
            *[blk(c2) for _ in range(num_refinement_blocks)]
        )
        self.output = nn.Conv2d(c2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
        else:
            latent = self.latent_reduce(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        return self.output(out_dec_level1) + inp_img
