"""
UNetCNN variant — encoder resnext101_32x16d.fb_swsl_ig1b_ft_in1k.

File độc lập với src/models/unetcnn.py (không sửa file cũ để các thí nghiệm
resnet101 trước đó không bị ảnh hưởng). Decoder CNN giữ nguyên kiến trúc
(Channel Projection Block + DecoderBlock upsample/concat/conv).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# resnext101_32x16d có cấu trúc stage giống ResNet (bottleneck), out_indices=(1,2,3,4)
_ENCODER_CHANNELS = {
    'resnext101_32x16d.fb_swsl_ig1b_ft_in1k': [256, 512, 1024, 2048],
}
_UNIFIED_CHANNELS = [64, 128, 256, 512]  # target dims after projection


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample x 2, concatenate skip, then 2x ConvBNReLU."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetCNN(nn.Module):
    """UNetCNN với encoder resnext101_32x16d.fb_swsl_ig1b_ft_in1k."""

    def __init__(self, encoder_name: str = 'resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
                 num_classes: int = 9, pretrained: bool = True):
        super().__init__()
        assert encoder_name in _ENCODER_CHANNELS, \
            f"encoder_name must be one of {list(_ENCODER_CHANNELS)}"

        enc_ch = _ENCODER_CHANNELS[encoder_name]

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            pretrained=pretrained,
        )

        # ── Channel Projection Block ────────────────────────────────────────
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch[i], _UNIFIED_CHANNELS[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(_UNIFIED_CHANNELS[i]),
                nn.ReLU(inplace=True),
            )
            for i in range(4)
        ])

        # ── CNN Decoder ─────────────────────────────────────────────────────
        self.dec3 = DecoderBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.dec1 = DecoderBlock(in_ch=128, skip_ch=64,  out_ch=64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        feats = self.encoder(x)
        p1, p2, p3, p4 = [self.proj[i](feats[i]) for i in range(4)]

        d3 = self.dec3(p4, p3)
        d2 = self.dec2(d3, p2)
        d1 = self.dec1(d2, p1)

        out = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=False)
        return self.classifier(out)


def build_model(cfg: dict) -> UNetCNN:
    return UNetCNN(
        encoder_name=cfg['MODEL']['ENCODER'],
        num_classes=cfg['TRAIN']['NUM_CLASSES'],
        pretrained=cfg['MODEL']['PRETRAINED'],
    )
