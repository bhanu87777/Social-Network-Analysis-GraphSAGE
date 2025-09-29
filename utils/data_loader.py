# utils/data_loader.py
import torch
from torch_geometric.datasets import Planetoid
import os
import random
import numpy as np
from config import SEED, DEVICE  # changed to absolute import

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_cora(root="data/cora"):
    set_seeds()
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    data = data.to(DEVICE)
    return dataset, data
