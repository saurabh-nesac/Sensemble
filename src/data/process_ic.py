def select_ic_fields(ds):
    return ds[["T", "QVAPOR", "U", "V", "PSFC"]]

def normalize(ds):
    return (ds - ds.mean("member")) / ds.std("member")
