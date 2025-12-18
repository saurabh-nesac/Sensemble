from src.data.dataset import WRFDAEnsembleICDataset
from src.models.generator import Generator
from src.models.discriminator import Discriminator
import torch

ds = WRFDAEnsembleICDataset(
    ens_dir="src/data/raw/wrfarw_ensmem_2025092200",
    conditioning_vars=["PSFC", "T2", "Q2"],
    target_vars=["T", "QVAPOR"],
    patch_size=64,
)

x, y = ds[0]
print(x.shape, y.shape)


G = Generator(in_channels=4, out_channels=2)  # e.g. 3 cond + 1 noise
x = torch.randn(8, 4, 64, 64)
y = G(x)

print(y.shape)


D = Discriminator(cond_channels=3, target_channels=2)
cond = torch.randn(8, 3, 64, 64)
tgt = torch.randn(8, 2, 64, 64)

out = D(cond, tgt)
print(out.shape)

