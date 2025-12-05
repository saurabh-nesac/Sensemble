import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from netCDF4 import Dataset as nc
import random
import matplotlib.pyplot as plt

# USER CONFIG

WRF_FILE = "/home/saurabh-nesac/Desktop/wrfout_d02_2025-10-07_00_00_00.nc"
VAR = "RAINNC"
PATCH = 64
BATCH = 8
EPOCHS = 30
LR = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


class WRFDataset(Dataset):
    def __init__(self, wrf_file, var, patch=64):
        ds = nc(wrf_file)
        rain = ds[var][:]  # shape [T, Y, X]
        ds.close()

        rain = rain.astype(np.float32)
        rain = rain - rain.min()
        rain = rain / (rain.max() + 1e-6)

        self.data = rain
        self.T, self.H, self.W = rain.shape

        self.patch = patch

    def __len__(self):
        return self.T * 10  # augment

    def __getitem__(self, idx):
        # pick timestep
        t = random.randint(0, self.T - 2)

        X = self.data[t]
        # temporal augmentation target
        Y = self.data[t + 1]

        # pick random crop
        y0 = random.randint(0, self.H - self.patch)
        x0 = random.randint(0, self.W - self.patch)

        Xp = X[y0:y0+self.patch, x0:x0+self.patch]
        Yp = Y[y0:y0+self.patch, x0:x0+self.patch]

        return torch.tensor(Xp).unsqueeze(0), torch.tensor(Yp).unsqueeze(0)

# ============================================================
# GENERATOR (U-Net Lite)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out): 
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = ConvBlock(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        br = self.bridge(p2)

        up2 = self.up2(br)
        cat2 = torch.cat([up2, c2], dim=1)
        d2 = self.dec2(cat2)

        up1 = self.up1(d2)
        cat1 = torch.cat([up1, c1], dim=1)
        d1 = self.dec1(cat1)

        return torch.sigmoid(self.out(d1))
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, True)
            )

        self.net = nn.Sequential(
            block(2, 32),
            block(32, 64),
            block(64, 128),
            nn.Conv2d(128, 1, 4, padding=1)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


ds = WRFDataset(WRF_FILE, VAR, PATCH)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

bce = nn.BCEWithLogitsLoss()
g_opt = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
d_opt = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

os.makedirs("samples", exist_ok=True)


for epoch in range(1, EPOCHS+1):
    pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}")

    for X, Y in pbar:
        X = X.to(device)
        Y = Y.to(device)

        #  Train Discriminator
        D.zero_grad()
        real_pred = D(X, Y)
        fake = G(X).detach()
        fake_pred = D(X, fake)

        d_real_loss = bce(real_pred, torch.ones_like(real_pred))
        d_fake_loss = bce(fake_pred, torch.zeros_like(fake_pred))
        d_loss = (d_real_loss + d_fake_loss) * 0.5

        d_loss.backward()
        d_opt.step()

        #  Train Generator
        G.zero_grad()
        fake = G(X)
        fake_pred = D(X, fake)

        g_loss = bce(fake_pred, torch.ones_like(fake_pred))
        g_loss.backward()
        g_opt.step()

        pbar.set_postfix({"D": d_loss.item(), "G": g_loss.item()})

    # Save sample
    with torch.no_grad():
        sample = fake[0].cpu().numpy()[0]
        print(sample)
        plt.imshow(sample, cmap="jet")
        plt.colorbar()
        plt.title(f"Epoch {epoch}")
        plt.savefig(f"samples/epoch_{epoch}.png")
        plt.close()

print("Training complete.")
