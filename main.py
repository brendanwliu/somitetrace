from model import unet
from omegaconf import DictConfig
import hydra

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

@hydra.main(config_path = "conf/config.yaml")

def my_app(cfg: DictConfig) -> None:
    print(cfg.pretty())
