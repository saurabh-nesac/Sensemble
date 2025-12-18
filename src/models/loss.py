import torch
import torch.nn as nn


class CGANLoss(nn.Module):
    """
    Loss wrapper for Conditional GAN (WRFDA ensemble ICs).

    Components:
        - Adversarial loss (PatchGAN, BCEWithLogits)
        - L1 reconstruction loss (mean / large-scale structure)

    Total Generator loss:
        L_G = L_adv + lambda_l1 * L1

    Discriminator loss:
        L_D = 0.5 * (L_real + L_fake)
    """

    def __init__(self, lambda_l1: float = 10.0):
        super().__init__()
        self.adv = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    # --------------------------------------------------------
    # Discriminator loss
    # --------------------------------------------------------
    def discriminator_loss(self, real_logits, fake_logits):
        """
        real_logits: D(cond, real)
        fake_logits: D(cond, fake.detach())
        """
        real_loss = self.adv(
            real_logits, torch.ones_like(real_logits)
        )
        fake_loss = self.adv(
            fake_logits, torch.zeros_like(fake_logits)
        )
        return 0.5 * (real_loss + fake_loss)

    # --------------------------------------------------------
    # Generator loss
    # --------------------------------------------------------
    def generator_loss(self, fake_logits, fake, real):
        """
        fake_logits: D(cond, fake)
        fake        : G(cond, z)
        real        : ground truth ensemble member
        """
        adv_loss = self.adv(
            fake_logits, torch.ones_like(fake_logits)
        )
        l1_loss = self.l1(fake, real)
        total = adv_loss + self.lambda_l1 * l1_loss

        return total, {
            "adv": adv_loss.item(),
            "l1": l1_loss.item(),
        }
