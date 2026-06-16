import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Encoder output channels per stage (out_indices=(1,2,3,4) for ResNets, (0,1,2,3) for MiT)
_ENCODER_CHANNELS = {
    'resnet18':  [64,  128,  256,  512],
    'resnet101': [256, 512, 1024, 2048],
    'mit_b0':    [32,   64,  160,  256],
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
    """Upsample × 2, concatenate skip, then 2× ConvBNReLU."""

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


class UNetFormer(nn.Module):
    """Multi-encoder UNetFormer with CNN decoder.

    Ablation encoder options: 'resnet18', 'resnet101', 'mit_b0'.
    Channel Projection Block normalises encoder outputs to [64, 128, 256, 512]
    so the CNN decoder is always the same regardless of encoder choice.
    """

    def __init__(self, encoder_name: str = 'resnet101', num_classes: int = 9,
                 pretrained: bool = True):
        super().__init__()
        assert encoder_name in _ENCODER_CHANNELS, \
            f"encoder_name must be one of {list(_ENCODER_CHANNELS)}"

        enc_ch = _ENCODER_CHANNELS[encoder_name]

        # ── Encoder ────────────────────────────────────────────────────────
        if encoder_name in ('resnet18', 'resnet101'):
            self.encoder = timm.create_model(
                encoder_name,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                pretrained=pretrained,
            )
        else:  # mit_b0
            self.encoder = timm.create_model(
                'mit_b0',
                features_only=True,
                out_indices=(0, 1, 2, 3),
                pretrained=pretrained,
            )

        # ── Channel Projection Block ────────────────────────────────────────
        # 4 × 1×1 Conv: enc_ch[i] → _UNIFIED_CHANNELS[i]
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch[i], _UNIFIED_CHANNELS[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(_UNIFIED_CHANNELS[i]),
                nn.ReLU(inplace=True),
            )
            for i in range(4)
        ])

        # ── CNN Decoder ─────────────────────────────────────────────────────
        # p4(512) → up → cat p3(256) → 256
        # → up → cat p2(128) → 128
        # → up → cat p1(64)  → 64
        # → final upsample 4× → classifier
        self.dec3 = DecoderBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.dec1 = DecoderBlock(in_ch=128, skip_ch=64,  out_ch=64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder: [s1, s2, s3, s4] at strides [4, 8, 16, 32]
        feats = self.encoder(x)

        # Channel projection → unified [64, 128, 256, 512]
        p1, p2, p3, p4 = [self.proj[i](feats[i]) for i in range(4)]

        # Decode bottom-up
        d3 = self.dec3(p4, p3)   # (B, 256, H/16, W/16)
        d2 = self.dec2(d3, p2)   # (B, 128, H/8,  W/8)
        d1 = self.dec1(d2, p1)   # (B,  64, H/4,  W/4)

        # Final upsample to input resolution
        out = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=False)
        return self.classifier(out)


def build_model(cfg: dict) -> UNetFormer:
    return UNetFormer(
        encoder_name=cfg['MODEL']['ENCODER'],
        num_classes=cfg['TRAIN']['NUM_CLASSES'],
        pretrained=cfg['MODEL']['PRETRAINED'],
    )
