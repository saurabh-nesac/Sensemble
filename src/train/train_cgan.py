# src/train/train_cgan.py

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import WRFDAEnsembleICDataset
from src.models.generator import Generator
from src.models.discriminator import Discriminator


# ============================================================
# Training entry point (called from main.py)
# ============================================================
def train(data_cfg, model_cfg, train_cfg):

    device = (
        "cuda"
        if train_cfg.get("device", "cuda") == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    # --------------------------------------------------------
    # Dataset & DataLoader
    # --------------------------------------------------------
    dataset = WRFDAEnsembleICDataset(
        ens_dir=data_cfg["ens_dir"],
        conditioning_vars=data_cfg["conditioning_vars"],
        target_vars=data_cfg["target_vars"],
        patch_size=data_cfg["patch_size"],
        normalize=data_cfg.get("normalize", "per_variable"),
    )

    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    Cc = len(data_cfg["conditioning_vars"])
    Ct = len(data_cfg["target_vars"])
    noise_ch = model_cfg.get("noise_channels", 1)

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------
    G = Generator(
        in_channels=Cc + noise_ch,
        out_channels=Ct,
        base_channels=model_cfg.get("gen_base_channels", 64),
    ).to(device)

    D = Discriminator(
        cond_channels=Cc,
        target_channels=Ct,
        base_channels=model_cfg.get("disc_base_channels", 64),
    ).to(device)

    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------
    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    lambda_l1 = model_cfg.get("lambda_l1", 10.0)

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------
    g_opt = torch.optim.Adam(
        G.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )
    d_opt = torch.optim.Adam(
        D.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )

    # --------------------------------------------------------
    # Output dirs
    # --------------------------------------------------------
    out_dir = Path(train_cfg.get("out_dir", "samples"))
    ckpt_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    epochs = train_cfg["epochs"]

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

        for cond, real in pbar:
            cond = cond.to(device)   # [B, Cc, H, W]
            real = real.to(device)   # [B, Ct, H, W]

            B, _, H, W = cond.shape

            # ==================================================
            # Train Discriminator
            # ==================================================
            z = torch.randn(B, noise_ch, H, W, device=device)
            fake = G(torch.cat([cond, z], dim=1)).detach()

            D.zero_grad()

            real_logits = D(cond, real)
            fake_logits = D(cond, fake)

            d_real = adv_loss(
                real_logits, torch.ones_like(real_logits)
            )
            d_fake = adv_loss(
                fake_logits, torch.zeros_like(fake_logits)
            )

            d_loss = 0.5 * (d_real + d_fake)
            d_loss.backward()
            d_opt.step()

            # ==================================================
            # Train Generator
            # ==================================================
            z = torch.randn(B, noise_ch, H, W, device=device)
            fake = G(torch.cat([cond, z], dim=1))

            G.zero_grad()
            fake_logits = D(cond, fake)

            g_adv = adv_loss(
                fake_logits, torch.ones_like(fake_logits)
            )
            g_l1 = l1_loss(fake, real)

            g_loss = g_adv + lambda_l1 * g_l1
            g_loss.backward()
            g_opt.step()

            pbar.set_postfix(
                D=f"{d_loss.item():.3f}",
                G=f"{g_loss.item():.3f}",
            )

        # ----------------------------------------------------
        # Save sample & checkpoint
        # ----------------------------------------------------
        if epoch % train_cfg.get("save_every", 5) == 0:
            _save_sample(
                cond, real, fake, img_dir, epoch
            )
            _save_checkpoint(
                G, D, g_opt, d_opt, ckpt_dir, epoch
            )

    print("[INFO] Training complete")


# ============================================================
# Helpers
# ============================================================
def _save_checkpoint(G, D, g_opt, d_opt, ckpt_dir, epoch):
    ckpt = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "g_opt": g_opt.state_dict(),
        "d_opt": d_opt.state_dict(),
        "epoch": epoch,
    }
    torch.save(
        ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt"
    )


def _save_sample(cond, real, fake, img_dir, epoch):
    """
    Save a quick-look PNG of first channel (e.g., T or QVAPOR)
    """
    import matplotlib.pyplot as plt

    c = cond[0, 0].detach().cpu()
    r = real[0, 0].detach().cpu()
    f = fake[0, 0].detach().cpu()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(
        axs, [c, r, f], ["Cond", "Real", "Fake"]
    ):
        im = ax.imshow(data, cmap="coolwarm")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(img_dir / f"epoch_{epoch:03d}.png")
    plt.close()
