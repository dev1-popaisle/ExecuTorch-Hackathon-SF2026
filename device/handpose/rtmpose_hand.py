"""Pure-PyTorch port of RTMPose-m (hand5, 256x256) from MMPose.

Module names mirror MMPose so the official checkpoint loads with strict=True:
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_KEYPOINTS = 21
INPUT_SIZE = 256
SIMCC_SPLIT_RATIO = 2.0
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
BN_EPS = 1e-5  # The hand config overrides CSPNeXt's norm_cfg with plain SyncBN.


class ConvModule(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=BN_EPS)
        self.activate = nn.SiLU()

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = ConvModule(cin, cin, k, stride, padding, groups=cin)
        self.pointwise_conv = ConvModule(cin, cout, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        return x * self.act(self.fc(self.global_avgpool(x)))


class CSPNeXtBlock(nn.Module):
    def __init__(self, cin, cout, add_identity, kernel_size=5):
        super().__init__()
        self.conv1 = ConvModule(cin, cout, 3, 1, 1)
        self.conv2 = DepthwiseSeparableConvModule(cout, cout, kernel_size, 1, kernel_size // 2)
        self.add_identity = add_identity and cin == cout

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, num_blocks, add_identity, expand_ratio=0.5):
        super().__init__()
        mid = int(cout * expand_ratio)
        self.main_conv = ConvModule(cin, mid, 1)
        self.short_conv = ConvModule(cin, mid, 1)
        self.final_conv = ConvModule(2 * mid, cout, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid, mid, add_identity) for _ in range(num_blocks)])
        self.attention = ChannelAttention(2 * mid)

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        return self.final_conv(self.attention(torch.cat((x_main, x_short), dim=1)))


class SPPBottleneck(nn.Module):
    def __init__(self, cin, cout, kernel_sizes=(5, 9, 13)):
        super().__init__()
        mid = cin // 2
        self.conv1 = ConvModule(cin, mid, 1)
        self.poolings = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in kernel_sizes])
        self.conv2 = ConvModule(mid * (len(kernel_sizes) + 1), cout, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [p(x) for p in self.poolings], dim=1))


class CSPNeXt(nn.Module):
    # in_channels, out_channels, num_blocks, add_identity, use_spp (arch P5)
    ARCH = [[64, 128, 3, True, False], [128, 256, 6, True, False],
            [256, 512, 6, True, False], [512, 1024, 3, False, True]]

    def __init__(self, deepen_factor=0.67, widen_factor=0.75):
        super().__init__()
        c0 = int(self.ARCH[0][0] * widen_factor)
        self.stem = nn.Sequential(
            ConvModule(3, c0 // 2, 3, 2, 1),
            ConvModule(c0 // 2, c0 // 2, 3, 1, 1),
            ConvModule(c0 // 2, c0, 3, 1, 1),
        )
        for i, (cin, cout, n, add_identity, use_spp) in enumerate(self.ARCH):
            cin, cout = int(cin * widen_factor), int(cout * widen_factor)
            n = max(round(n * deepen_factor), 1)
            stage = [ConvModule(cin, cout, 3, 2, 1)]
            if use_spp:
                stage.append(SPPBottleneck(cout, cout))
            stage.append(CSPLayer(cout, cout, n, add_identity))
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))

    def forward(self, x):
        x = self.stem(x)
        for i in range(1, 5):
            x = getattr(self, f"stage{i}")(x)
        return x


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Explicit sqrt-of-sum instead of torch.norm so every backend lowers it.
        norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True)) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class RTMCCBlock(nn.Module):
    """Gated Attention Unit, self-attention, no relative bias, no RoPE."""

    def __init__(self, dims=256, expansion_factor=2, s=128):
        super().__init__()
        self.s = s
        self.e = int(dims * expansion_factor)
        self.o = nn.Linear(self.e, dims, bias=False)
        self.uv = nn.Linear(dims, 2 * self.e + s, bias=False)
        self.gamma = nn.Parameter(torch.rand((2, s)))
        self.beta = nn.Parameter(torch.rand((2, s)))
        self.ln = ScaleNorm(dims)
        self.act_fn = nn.SiLU()
        self.res_scale = Scale(dims)
        self.sqrt_s = math.sqrt(s)

    def forward(self, x):
        uv = self.act_fn(self.uv(self.ln(x)))
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        q, k = base[:, :, 0], base[:, :, 1]
        kernel = torch.square(F.relu(torch.bmm(q, k.permute(0, 2, 1)) / self.sqrt_s))
        out = self.o(u * torch.bmm(kernel, v))
        return self.res_scale(x) + out


class RTMCCHead(nn.Module):
    def __init__(self, in_channels=768, out_channels=NUM_KEYPOINTS, input_size=INPUT_SIZE,
                 featuremap=INPUT_SIZE // 32, hidden=256, split_ratio=SIMCC_SPLIT_RATIO):
        super().__init__()
        self.final_layer = nn.Conv2d(in_channels, out_channels, 7, 1, 3)
        self.mlp = nn.Sequential(ScaleNorm(featuremap * featuremap),
                                 nn.Linear(featuremap * featuremap, hidden, bias=False))
        self.gau = RTMCCBlock(hidden)
        bins = int(input_size * split_ratio)
        self.cls_x = nn.Linear(hidden, bins, bias=False)
        self.cls_y = nn.Linear(hidden, bins, bias=False)

    def forward(self, feats):
        x = torch.flatten(self.final_layer(feats), 2)
        x = self.gau(self.mlp(x))
        return self.cls_x(x), self.cls_y(x)


class RTMPoseHand(nn.Module):
    """Input: RGB float (B,3,256,256) in [0,1]. Output: simcc_x, simcc_y each (B,21,512)."""

    def __init__(self):
        super().__init__()
        self.backbone = CSPNeXt()
        self.head = RTMCCHead()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) / 255.0, persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) / 255.0, persistent=False)

    def forward(self, image):
        return self.head(self.backbone((image - self.mean) / self.std))

    def load_mmpose_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        sd = {k: v for k, v in sd.items() if k.startswith(("backbone.", "head."))}
        self.load_state_dict(sd, strict=True)
        return self


def decode_simcc(simcc_x: torch.Tensor, simcc_y: torch.Tensor):
    """Return keypoints (B,21,2) in 256x256 input pixels and scores (B,21)."""
    max_x, ix = simcc_x.max(dim=-1)
    max_y, iy = simcc_y.max(dim=-1)
    kpts = torch.stack([ix, iy], dim=-1).float() / SIMCC_SPLIT_RATIO
    return kpts, torch.minimum(max_x, max_y)
