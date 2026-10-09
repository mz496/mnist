import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)                 # Python's built-in RNG
    np.random.seed(seed)              # NumPy's global RNG
    torch.manual_seed(seed)           # PyTorch CPU RNG (also seeds CUDA in recent versions)
    # torch.cuda.manual_seed_all(seed)  # all GPUs, to be explicit - still needed?