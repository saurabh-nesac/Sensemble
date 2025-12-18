import torch
import torch.nn as nn


# ============================================================
# Basic discriminator block
# ============================================================
class DiscBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not norm,
            )
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ============================================================
# Conditional PatchGAN Discriminator
# ============================================================
class Discriminator(nn.Module):
    """
    Conditional discriminator for WRFDA ensemble ICs.

    Input:
        cond   : conditioning fields  [B, Cc, H, W]
        target : real/fake fields     [B, Ct, H, W]

    Output:
        Patch-wise realism score      [B, 1, H', W']
    """

    def __init__(
        self,
        cond_channels: int,
        target_channels: int,
        base_channels: int = 64,
    ):
        super().__init__()

        in_channels = cond_channels + target_channels

        self.net = nn.Sequential(
            # No norm in first layer (PatchGAN standard)
            DiscBlock(in_channels, base_channels, norm=False),

            DiscBlock(base_channels, base_channels * 2),
            DiscBlock(base_channels * 2, base_channels * 4),

            # Final conv → patch realism map
            nn.Conv2d(
                base_channels * 4,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )

    def forward(self, cond, target):
        """
        cond   : [B, Cc, H, W]
        target : [B, Ct, H, W]
        """
        x = torch.cat([cond, target], dim=1)
        return self.net(x)
