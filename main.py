import yaml
from src.train.train_cgan import train

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    data_cfg  = load_cfg("config/data.yaml")
    model_cfg = load_cfg("config/model.yaml")
    train_cfg = load_cfg("config/train.yaml")

    train(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg
    )
