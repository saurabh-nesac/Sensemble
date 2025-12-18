from pathlib import Path
import xarray as xr
import numpy as np

def load_wrfda_ensemble(ens_dir: str) -> xr.Dataset:
    files = sorted(Path(ens_dir).glob("wrfarw.mem*"))

    datasets = []
    for i, f in enumerate(files, start=1):
        ds = xr.open_dataset(f)
        ds = ds.expand_dims(member=[i])
        datasets.append(ds)

    ens = xr.concat(datasets, dim="member")
    return ens
