from src.data.dataset import WRFDAEnsembleICDataset

ds = WRFDAEnsembleICDataset(
    ens_dir="src/data/raw/wrfarw_ensmem_2025092200",
    conditioning_vars=["PSFC", "T2", "Q2"],
    target_vars=["T", "QVAPOR"],
    patch_size=64,
)

x, y = ds[0]
print(x.shape, y.shape)
