import torch
import torch.nn as nn


# ============================================================
# Basic building block
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Generator (U-Net Lite)
# ============================================================
class Generator(nn.Module):
    """
    Conditional Generator for WRFDA ensemble ICs

    Input:
        [conditioning vars | white noise]
        shape = (B, Cc + Cz, H, W)

    Output:
        target vars
        shape = (B, Ct, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
    ):
        super().__init__()

        # ---------------- Encoder ----------------
        self.down1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        # ---------------- Bottleneck ----------------
        self.bridge = ConvBlock(base_channels * 2, base_channels * 4)

        # ---------------- Decoder ----------------
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # ---------------- Output ----------------
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        # Bottleneck
        b = self.bridge(p2)

        # Decoder
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        # Output
        return self.out(d1)
